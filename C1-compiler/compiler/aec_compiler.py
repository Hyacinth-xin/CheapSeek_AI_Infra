#!/usr/bin/env python3
"""C1 PTX 9.3 subset to AEC scalar ISA compiler.

The implementation is deliberately self-contained and uses only Python's
standard library so it can run in the offline judging container.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


class CompileError(Exception):
    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


class SpillNeeded(Exception):
    def __init__(self, virtual_register: str):
        self.virtual_register = virtual_register
        super().__init__(virtual_register)


TYPE_CODES = {
    "b32": 0x0,
    "b64": 0x1,
    "u32": 0x2,
    "s32": 0x3,
    "u64": 0x1,
    "f32": 0x8,
    "none": 0xF,
}

TYPE_SIZE_ALIGN = {
    "b32": (4, 4),
    "u32": (4, 4),
    "s32": (4, 4),
    "f32": (4, 4),
    "b64": (8, 8),
    "u64": (8, 8),
}

OPCODES = {
    "ADD": 0x0001,
    "SUB": 0x0002,
    "MUL": 0x0003,
    "MAD": 0x0004,
    "FMA": 0x0005,
    "AND": 0x0010,
    "OR": 0x0011,
    "XOR": 0x0012,
    "SHL": 0x0014,
    "SHR": 0x0015,
    "CMPP": 0x0021,
    "LD": 0x0030,
    "ST": 0x0031,
    "BR": 0x0040,
    "BRX": 0x0041,
    "HALT": 0x0045,
    "CPY": 0x0054,
    "LOADI": 0x0055,
    "LOADI64": 0x0056,
}

COMPARE_SUBOPS = {"eq": 0, "ne": 1, "lt": 2, "le": 3, "gt": 4, "ge": 5}
SPACE_CODES = {"gmem": 0, "smem": 1, "cmem": 2, "lmem": 3, "pmem": 4}

SPECIAL_REGS = {
    "%tid.x": 0x0100,
    "%ntid.x": 0x0101,
    "%ctaid.x": 0x0102,
    "%nctaid.x": 0x0103,
    "%laneid": 0x0104,
    "%tid.y": 0x0110,
    "%ntid.y": 0x0111,
    "%ctaid.y": 0x0112,
    "%nctaid.y": 0x0113,
    "%tid.z": 0x0120,
    "%ntid.z": 0x0121,
    "%ctaid.z": 0x0122,
    "%nctaid.z": 0x0123,
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def split_operands(text: str) -> List[str]:
    result: List[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in "[({<":
            depth += 1
        elif ch in "])}>" and depth:
            depth -= 1
        elif ch == "," and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result


def parse_int(token: str, line: Optional[int] = None) -> int:
    token = token.strip()
    # PTX permits an optional U/L suffix on integer literals.
    token = re.sub(r"(?i)(ull|llu|ul|lu|u|ll|l)$", "", token)
    try:
        return int(token, 0)
    except ValueError as exc:
        raise CompileError(f"invalid integer immediate '{token}'", line) from exc


def is_register(token: str) -> bool:
    return token.startswith("%") and token not in SPECIAL_REGS


def memory_inner(token: str) -> str:
    token = token.strip()
    if not (token.startswith("[") and token.endswith("]")):
        raise CompileError(f"expected memory operand, got '{token}'")
    return token[1:-1].strip()


@dataclass
class Parameter:
    name: str
    typ: str
    offset: int


@dataclass
class PTXInstruction:
    mnemonic: str
    operands: List[str]
    line: int
    guard: Optional[str] = None
    guard_neg: bool = False

    def clone(self) -> "PTXInstruction":
        return PTXInstruction(self.mnemonic, list(self.operands), self.line, self.guard, self.guard_neg)


@dataclass
class PTXBlock:
    name: str
    instructions: List[PTXInstruction] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)


@dataclass
class PTXProgram:
    kernel: str
    params: List[Parameter]
    register_types: Dict[str, str]
    blocks: List[PTXBlock]
    num_source_instructions: int


@dataclass(frozen=True)
class Special:
    selector: int


Operand = Union[str, Special, None]


@dataclass
class MIRInstruction:
    op: str
    typ: str = "none"
    dest: Optional[str] = None
    src1: Operand = None
    src2: Operand = None
    src3: Operand = None
    imm: Optional[int] = None
    space: Optional[str] = None
    subop: int = 0
    target: Optional[str] = None
    pred: Optional[str] = None
    pred_neg: bool = False
    line: int = 0
    pair_def: bool = False
    pair_src: bool = False

    def gpr_uses(self, pair_hi: Dict[str, str]) -> Set[str]:
        uses: Set[str] = set()
        for source in (self.src1, self.src2, self.src3):
            if isinstance(source, str):
                uses.add(source)
        if self.pair_src and isinstance(self.src1, str):
            high = pair_hi.get(self.src1)
            if high:
                uses.add(high)
        return uses

    def gpr_defs(self, pair_hi: Dict[str, str]) -> Set[str]:
        if not self.dest or self.op == "CMPP":
            return set()
        result = {self.dest}
        if self.pair_def:
            high = pair_hi.get(self.dest)
            if high:
                result.add(high)
        return result

    def pred_uses(self) -> Set[str]:
        return {self.pred} if self.pred else set()

    def pred_defs(self) -> Set[str]:
        return {self.dest} if self.op == "CMPP" and self.dest else set()


@dataclass
class MIRBlock:
    name: str
    instructions: List[MIRInstruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)


@dataclass
class MIRProgram:
    kernel: str
    blocks: List[MIRBlock]
    pair_hi: Dict[str, str]
    pair_groups: Dict[str, Tuple[str, str]]
    num_virtual_registers: int


def parse_ptx(text: str) -> PTXProgram:
    cleaned = strip_comments(text)
    version = re.search(r"(?m)^\s*\.version\s+([^\s]+)", cleaned)
    target = re.search(r"(?m)^\s*\.target\s+([^\s]+)", cleaned)
    address_size = re.search(r"(?m)^\s*\.address_size\s+(\d+)", cleaned)
    if not version or version.group(1) != "9.3":
        raise CompileError("C1 requires '.version 9.3'")
    if not target or target.group(1) != "sm_90":
        raise CompileError("C1 requires '.target sm_90'")
    if not address_size or address_size.group(1) != "64":
        raise CompileError("C1 requires '.address_size 64'")

    entry = re.search(
        r"\.visible\s+\.entry\s+([A-Za-z_.$][\w.$]*)\s*\((.*?)\)\s*\{(.*)\}\s*$",
        cleaned,
        flags=re.S,
    )
    if not entry:
        raise CompileError("expected one '.visible .entry' kernel")
    kernel, param_text, body = entry.group(1), entry.group(2), entry.group(3)

    params: List[Parameter] = []
    offset = 0
    for raw in split_operands(param_text):
        match = re.fullmatch(r"\s*\.param\s+\.(u32|s32|u64|b32|b64|f32)\s+([A-Za-z_.$][\w.$]*)\s*", raw)
        if not match:
            raise CompileError(f"invalid kernel parameter declaration '{raw}'")
        typ, name = match.group(1), match.group(2)
        size, alignment = TYPE_SIZE_ALIGN[typ]
        offset = align_up(offset, alignment)
        params.append(Parameter(name, typ, offset))
        offset += size
    _param_block_size = align_up(offset, 8)

    register_types: Dict[str, str] = {}
    items: List[Tuple[str, Union[str, PTXInstruction]]] = []
    pending = ""
    pending_line = 0
    body_start_line = cleaned[: entry.start(3)].count("\n") + 1

    def parse_statement(statement: str, line: int) -> None:
        statement = statement.strip()
        if not statement:
            return
        while True:
            inline_label = re.match(r"^([A-Za-z_.$][\w.$]*):\s*", statement)
            if not inline_label:
                break
            items.append(("label", inline_label.group(1)))
            statement = statement[inline_label.end() :].strip()
            if not statement:
                return
        reg = re.fullmatch(r"\.reg\s+\.(pred|u32|s32|u64|b32|b64|f32)\s+(%[A-Za-z_$][\w$]*)<(\d+)>", statement)
        if reg:
            typ, base, count_s = reg.group(1), reg.group(2), reg.group(3)
            count = int(count_s)
            for i in range(count):
                register_types[f"{base}{i}"] = typ
            return
        guard = None
        guard_neg = False
        guard_match = re.match(r"@(!?)(%[A-Za-z_$][\w$]*)\s+", statement)
        if guard_match:
            guard_neg = guard_match.group(1) == "!"
            guard = guard_match.group(2)
            statement = statement[guard_match.end() :].strip()
        parts = statement.split(None, 1)
        mnemonic = parts[0].lower()
        operands = split_operands(parts[1]) if len(parts) == 2 else []
        items.append(("inst", PTXInstruction(mnemonic, operands, line, guard, guard_neg)))

    for relative_line, raw_line in enumerate(body.splitlines(), start=0):
        line_no = body_start_line + relative_line
        work = raw_line.strip()
        if not work:
            continue
        while True:
            label_match = re.match(r"^([A-Za-z_.$][\w.$]*):", work)
            if not label_match:
                break
            if pending.strip():
                raise CompileError("label encountered in the middle of an instruction", line_no)
            items.append(("label", label_match.group(1)))
            work = work[label_match.end() :].strip()
            if not work:
                break
        if not work:
            continue
        if not pending:
            pending_line = line_no
        pending = f"{pending} {work}".strip()
        while ";" in pending:
            statement, pending = pending.split(";", 1)
            parse_statement(statement, pending_line)
            pending = pending.strip()
            pending_line = line_no
    if pending.strip():
        raise CompileError(f"unterminated statement '{pending.strip()}'", pending_line)

    source_count = sum(1 for kind, _ in items if kind == "inst")
    blocks: List[PTXBlock] = []
    current: Optional[PTXBlock] = None
    synthetic_id = 0

    def start_block(name: Optional[str] = None) -> PTXBlock:
        nonlocal synthetic_id
        if name is None:
            name = f"__bb{synthetic_id}"
            synthetic_id += 1
        block = PTXBlock(name=name, aliases=[name])
        blocks.append(block)
        return block

    for kind, value in items:
        if kind == "label":
            label = str(value)
            if current is None or current.instructions:
                current = start_block(label)
            else:
                current.aliases.append(label)
            continue
        inst = value
        assert isinstance(inst, PTXInstruction)
        if current is None:
            current = start_block("__entry" if not blocks else None)
        current.instructions.append(inst)
        if inst.mnemonic == "bra" or inst.mnemonic == "ret":
            current = None
    if not blocks:
        raise CompileError("kernel body contains no instructions")

    alias_to_block: Dict[str, str] = {}
    for block in blocks:
        for alias in block.aliases:
            if alias in alias_to_block:
                raise CompileError(f"duplicate label '{alias}'")
            alias_to_block[alias] = block.name

    for i, block in enumerate(blocks):
        if not block.instructions:
            if i + 1 < len(blocks):
                block.successors = [blocks[i + 1].name]
            continue
        tail = block.instructions[-1]
        if tail.mnemonic == "ret":
            block.successors = []
        elif tail.mnemonic == "bra":
            if len(tail.operands) != 1:
                raise CompileError("bra requires one label operand", tail.line)
            target = tail.operands[0]
            if target not in alias_to_block:
                raise CompileError(f"undefined branch label '{target}'", tail.line)
            tail.operands[0] = alias_to_block[target]
            block.successors = [alias_to_block[target]]
            if tail.guard and i + 1 < len(blocks):
                block.successors.append(blocks[i + 1].name)
        elif i + 1 < len(blocks):
            block.successors = [blocks[i + 1].name]

    validate_ptx(PTXProgram(kernel, params, register_types, blocks, source_count))
    return PTXProgram(kernel, params, register_types, blocks, source_count)


def validate_ptx(program: PTXProgram) -> None:
    allowed_exact = {
        "ret",
        "bra",
        "add.u32",
        "sub.u32",
        "mul.lo.u32",
        "mad.lo.u32",
        "mul.wide.u32",
        "add.u64",
        "and.b32",
        "or.b32",
        "xor.b32",
        "shl.b32",
        "shr.u32",
        "add.f32",
        "add.rn.f32",
        "sub.f32",
        "sub.rn.f32",
        "mul.f32",
        "mul.rn.f32",
        "mad.f32",
        "mad.rn.f32",
        "fma.rn.f32",
    }
    patterns = [
        r"ld\.param\.(u32|u64|b32|b64)",
        r"mov\.(u32|u64|b32|b64)",
        r"setp\.(eq|ne|lt|le|gt|ge)\.u32",
        r"ld\.global\.(f32|u32|b32)",
        r"st\.global\.(f32|u32|b32)",
    ]
    param_names = {p.name for p in program.params}
    for block in program.blocks:
        for inst in block.instructions:
            if inst.mnemonic not in allowed_exact and not any(re.fullmatch(p, inst.mnemonic) for p in patterns):
                raise CompileError(f"unsupported PTX instruction '{inst.mnemonic}'", inst.line)
            if inst.guard and inst.mnemonic != "bra":
                raise CompileError("the C1 subset only permits predication on bra", inst.line)
            if inst.guard and program.register_types.get(inst.guard) != "pred":
                raise CompileError(f"unknown predicate register '{inst.guard}'", inst.line)
            for operand in inst.operands:
                raw = operand.strip()
                inner = raw[1:-1].strip() if raw.startswith("[") and raw.endswith("]") else raw
                if inner.startswith("%") and inner not in SPECIAL_REGS and inner not in program.register_types:
                    raise CompileError(f"undeclared register '{inner}'", inst.line)
            if inst.mnemonic.startswith("ld.param"):
                if len(inst.operands) != 2 or memory_inner(inst.operands[1]) not in param_names:
                    raise CompileError("ld.param references an unknown parameter", inst.line)


def instruction_defs_uses(inst: PTXInstruction) -> Tuple[Set[str], Set[str]]:
    defs: Set[str] = set()
    uses: Set[str] = set()
    mnemonic = inst.mnemonic
    if mnemonic not in {"bra", "ret"} and not mnemonic.startswith("st.") and inst.operands:
        if is_register(inst.operands[0]):
            defs.add(inst.operands[0])
    start = 0 if mnemonic.startswith("st.") else 1
    for operand in inst.operands[start:]:
        token = operand.strip()
        if token.startswith("[") and token.endswith("]"):
            token = token[1:-1].strip()
        if is_register(token):
            uses.add(token)
    if inst.guard:
        uses.add(inst.guard)
    return defs, uses


def is_pure(inst: PTXInstruction) -> bool:
    return inst.mnemonic not in {"bra", "ret"} and not inst.mnemonic.startswith("st.")


def dce(program: PTXProgram) -> int:
    """Global liveness-based dead code elimination on mutable PTX registers."""
    block_map = {b.name: b for b in program.blocks}
    removed_total = 0
    changed = True
    while changed:
        changed = False
        use: Dict[str, Set[str]] = {}
        defs: Dict[str, Set[str]] = {}
        for block in program.blocks:
            b_use: Set[str] = set()
            b_def: Set[str] = set()
            for inst in block.instructions:
                idef, iuse = instruction_defs_uses(inst)
                b_use.update(iuse - b_def)
                b_def.update(idef)
            use[block.name], defs[block.name] = b_use, b_def
        live_in = {b.name: set() for b in program.blocks}
        live_out = {b.name: set() for b in program.blocks}
        progress = True
        while progress:
            progress = False
            for block in reversed(program.blocks):
                out = set().union(*(live_in[s] for s in block.successors if s in block_map)) if block.successors else set()
                inn = use[block.name] | (out - defs[block.name])
                if out != live_out[block.name] or inn != live_in[block.name]:
                    live_out[block.name], live_in[block.name] = out, inn
                    progress = True
        for block in program.blocks:
            live = set(live_out[block.name])
            kept: List[PTXInstruction] = []
            for inst in reversed(block.instructions):
                idef, iuse = instruction_defs_uses(inst)
                if idef and is_pure(inst) and idef.isdisjoint(live):
                    removed_total += 1
                    changed = True
                    continue
                live.difference_update(idef)
                live.update(iuse)
                kept.append(inst)
            block.instructions = list(reversed(kept))
    return removed_total


def merge_blocks(program: PTXProgram) -> int:
    """Merge consecutive basic blocks connected by a unique fall-through edge.

    Merging increases the scope of local CSE and the instruction scheduler,
    and eliminates unnecessary branches.  Only blocks whose sole predecessor
    is the immediately preceding block are merged so that the CFG remains
    well-formed for downstream passes.
    """
    if len(program.blocks) < 2:
        return 0

    pred_counts: Dict[str, int] = {}
    for block in program.blocks:
        pred_counts.setdefault(block.name, 0)
        for succ in block.successors:
            pred_counts[succ] = pred_counts.get(succ, 0) + 1

    merged = 0
    new_blocks: List[PTXBlock] = []
    skip: Set[str] = set()

    for i, block in enumerate(program.blocks):
        if block.name in skip:
            continue

        while len(block.successors) == 1:
            succ_name = block.successors[0]
            if pred_counts.get(succ_name, 0) != 1:
                break

            # Never merge when the block ends with an explicit terminator.
            # Deleting a bra and replacing it with fall-through is unsafe
            # because the bra target may not be the physically adjacent block.
            if not block.instructions or block.instructions[-1].mnemonic in {"bra", "ret"}:
                break

            succ_block = next((b for b in program.blocks if b.name == succ_name), None)
            if succ_block is None or succ_block.name in skip:
                break

            block.instructions.extend(succ_block.instructions)
            block.successors = list(succ_block.successors)
            for alias in succ_block.aliases:
                if alias not in block.aliases:
                    block.aliases.append(alias)

            skip.add(succ_name)
            merged += 1

        new_blocks.append(block)

    if merged:
        program.blocks = new_blocks
    return merged


def replace_source(operand: str, copies: Dict[str, str]) -> str:
    bracketed = operand.startswith("[") and operand.endswith("]")
    token = operand[1:-1].strip() if bracketed else operand
    seen: Set[str] = set()
    while token in copies and token not in seen:
        seen.add(token)
        token = copies[token]
    return f"[{token}]" if bracketed else token


def mov_mnemonic_for(inst: PTXInstruction) -> str:
    suffix = inst.mnemonic.split(".")[-1]
    if suffix == "f32":
        return "mov.b32"
    if suffix in {"u32", "s32", "b32"}:
        return "mov.b32" if suffix == "s32" else f"mov.{suffix}"
    if suffix in {"u64", "b64"}:
        return f"mov.{suffix}"
    return "mov.b32"


def local_cse(program: PTXProgram) -> int:
    optimized = 0
    for block in program.blocks:
        copies: Dict[str, str] = {}
        expressions: Dict[Tuple[object, ...], str] = {}
        new_insts: List[PTXInstruction] = []
        for original in block.instructions:
            inst = original.clone()
            defs, _ = instruction_defs_uses(inst)
            dest = next(iter(defs), None)
            source_start = 0 if inst.mnemonic.startswith("st.") else 1
            for i in range(source_start, len(inst.operands)):
                inst.operands[i] = replace_source(inst.operands[i], copies)

            if dest:
                copies.pop(dest, None)
                for key, value in list(copies.items()):
                    if value == dest:
                        copies.pop(key, None)
                for key, holder in list(expressions.items()):
                    mentions_dest = any(
                        part == dest
                        or (isinstance(part, str) and part.startswith("[") and part.endswith("]") and part[1:-1].strip() == dest)
                        for part in key
                    )
                    if holder == dest or mentions_dest:
                        expressions.pop(key, None)

            if inst.mnemonic.startswith("st."):
                expressions = {key: value for key, value in expressions.items() if not str(key[0]).startswith("ld.global")}

            cse_eligible = (
                dest is not None
                and inst.mnemonic not in {"bra", "ret"}
                and not inst.mnemonic.startswith("st.")
                and not inst.mnemonic.startswith("mov.")
                and not inst.mnemonic.startswith("setp.")
            )
            if cse_eligible:
                # Non-SSA safety: a self-updating instruction like
                #   sub.u32 %r1, %r1, %r2
                # cannot be reused because its result depends on the *old*
                # value of the destination register.
                defs, _ = instruction_defs_uses(inst)
                uses_dest = False
                for operand in inst.operands[1:]:
                    token = operand.strip()
                    if token.startswith("[") and token.endswith("]"):
                        token = token[1:-1].strip()
                    if token == dest:
                        uses_dest = True
                        break
                if uses_dest:
                    cse_eligible = False
            if cse_eligible:
                key = (inst.mnemonic, *inst.operands[1:])
                holder = expressions.get(key)
                if holder and holder != dest:
                    inst = PTXInstruction(mov_mnemonic_for(inst), [dest, holder], inst.line)
                    copies[dest] = holder
                    optimized += 1
                else:
                    expressions[key] = dest
            elif dest and inst.mnemonic.startswith("mov.") and len(inst.operands) == 2 and is_register(inst.operands[1]):
                copies[dest] = inst.operands[1]
            new_insts.append(inst)
        block.instructions = new_insts
    return optimized


def fuse_mad(program: PTXProgram) -> int:
    all_uses: Dict[str, int] = {}
    for block in program.blocks:
        for inst in block.instructions:
            _, uses = instruction_defs_uses(inst)
            for reg in uses:
                all_uses[reg] = all_uses.get(reg, 0) + 1
    fused = 0
    for block in program.blocks:
        out: List[PTXInstruction] = []
        i = 0
        while i < len(block.instructions):
            if i + 1 < len(block.instructions):
                mul = block.instructions[i]
                add = block.instructions[i + 1]
                if (
                    mul.mnemonic in {"mul.f32", "mul.rn.f32"}
                    and add.mnemonic in {"add.f32", "add.rn.f32"}
                    and len(mul.operands) == 3
                    and len(add.operands) == 3
                    and all_uses.get(mul.operands[0], 0) == 1
                    and mul.operands[0] in add.operands[1:]
                    and sum(1 for op in add.operands[1:] if op == mul.operands[0]) == 1
                ):
                    other = add.operands[2] if add.operands[1] == mul.operands[0] else add.operands[1]
                    out.append(PTXInstruction("mad.f32", [add.operands[0], mul.operands[1], mul.operands[2], other], add.line))
                    fused += 1
                    i += 2
                    continue
            out.append(block.instructions[i])
            i += 1
        block.instructions = out
    return fused


def licm(program: PTXProgram) -> int:
    """Conservative loop-invariant code motion for the mutable PTX IR."""
    if len(program.blocks) < 2:
        return 0
    block_by_name = {block.name: block for block in program.blocks}
    names = set(block_by_name)
    predecessors: Dict[str, Set[str]] = {name: set() for name in names}
    for block in program.blocks:
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(block.name)

    entry = program.blocks[0].name
    dominators: Dict[str, Set[str]] = {name: set(names) for name in names}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in program.blocks[1:]:
            preds = predecessors[block.name]
            if preds:
                new = {block.name} | set.intersection(*(dominators[p] for p in preds))
            else:
                new = {block.name}
            if new != dominators[block.name]:
                dominators[block.name] = new
                changed = True

    moved_total = 0
    for latch in list(program.blocks):
        for header_name in list(latch.successors):
            if header_name not in dominators[latch.name]:
                continue
            header = block_by_name[header_name]
            loop_nodes = {header.name, latch.name}
            stack = [latch.name]
            while stack:
                node = stack.pop()
                for pred in predecessors[node]:
                    if pred not in loop_nodes:
                        loop_nodes.add(pred)
                        stack.append(pred)
            outside = [p for p in predecessors[header.name] if p not in loop_nodes]
            if len(outside) != 1:
                continue
            preheader = block_by_name[outside[0]]
            if preheader.successors != [header.name]:
                continue

            loop_has_global_stores = any(
                inst2.mnemonic.startswith("st.global")
                for block in program.blocks
                if block.name in loop_nodes
                for inst2 in block.instructions
            )

            def_count: Dict[str, int] = {}
            outside_uses: Set[str] = set()
            use_sites: Dict[str, List[Tuple[str, int]]] = {}
            for block in program.blocks:
                for index, inst in enumerate(block.instructions):
                    defs, uses = instruction_defs_uses(inst)
                    if block.name in loop_nodes:
                        for defined in defs:
                            def_count[defined] = def_count.get(defined, 0) + 1
                        for used in uses:
                            use_sites.setdefault(used, []).append((block.name, index))
                    else:
                        outside_uses.update(uses)

            invariant_defs: Set[str] = set()
            selected: List[Tuple[PTXBlock, PTXInstruction]] = []
            progress = True
            while progress:
                progress = False
                for block in program.blocks:
                    if block.name not in loop_nodes:
                        continue
                    for index, inst in enumerate(block.instructions):
                        if any(inst is chosen for _, chosen in selected):
                            continue
                        defs, uses = instruction_defs_uses(inst)
                        if len(defs) != 1 or not is_pure(inst):
                            continue
                        dest = next(iter(defs))
                        if def_count.get(dest) != 1 or dest in outside_uses:
                            continue
                        if inst.mnemonic.startswith("setp."):
                            continue
                        if inst.mnemonic.startswith("ld.global") and loop_has_global_stores:
                            continue
                        if any(used in def_count and used not in invariant_defs for used in uses):
                            continue
                        dominates_uses = True
                        for use_block, use_index in use_sites.get(dest, []):
                            if use_block == block.name:
                                if index >= use_index:
                                    dominates_uses = False
                                    break
                            elif block.name not in dominators[use_block]:
                                dominates_uses = False
                                break
                        if not dominates_uses:
                            continue
                        selected.append((block, inst))
                        invariant_defs.add(dest)
                        progress = True
            if not selected:
                continue
            insertion = len(preheader.instructions)
            if insertion and preheader.instructions[-1].mnemonic in {"bra", "ret"}:
                insertion -= 1
            for block, inst in selected:
                for index, candidate in enumerate(block.instructions):
                    if candidate is inst:
                        block.instructions.pop(index)
                        break
            preheader.instructions[insertion:insertion] = [inst for _, inst in selected]
            moved_total += len(selected)
    return moved_total


def fold_integer_constants(program: PTXProgram) -> int:
    folded = 0
    binary = {
        "add.u32": lambda a, b: a + b,
        "sub.u32": lambda a, b: a - b,
        "mul.lo.u32": lambda a, b: a * b,
        "and.b32": lambda a, b: a & b,
        "or.b32": lambda a, b: a | b,
        "xor.b32": lambda a, b: a ^ b,
        "shl.b32": lambda a, b: a << (b & 31),
        "shr.u32": lambda a, b: (a & 0xFFFFFFFF) >> (b & 31),
    }
    for block in program.blocks:
        constants: Dict[str, int] = {}
        new_insts: List[PTXInstruction] = []
        for original in block.instructions:
            inst = original.clone()
            defs, _ = instruction_defs_uses(inst)
            dest = next(iter(defs), None)
            if dest:
                constants.pop(dest, None)

            def value_of(token: str) -> Optional[int]:
                if token in constants:
                    return constants[token]
                if is_register(token) or token in SPECIAL_REGS or token.startswith("["):
                    return None
                try:
                    return parse_int(token, inst.line) & 0xFFFFFFFF
                except CompileError:
                    return None

            replacement: Optional[PTXInstruction] = None
            if inst.mnemonic in binary and len(inst.operands) == 3:
                a, b = value_of(inst.operands[1]), value_of(inst.operands[2])
                if a is not None and b is not None:
                    value = binary[inst.mnemonic](a, b) & 0xFFFFFFFF
                    replacement = PTXInstruction("mov.u32", [inst.operands[0], str(value)], inst.line)
            elif inst.mnemonic == "mad.lo.u32" and len(inst.operands) == 4:
                values = [value_of(token) for token in inst.operands[1:]]
                if all(value is not None for value in values):
                    a, b, c = (int(value) for value in values)
                    replacement = PTXInstruction("mov.u32", [inst.operands[0], str((a * b + c) & 0xFFFFFFFF)], inst.line)
            if replacement is not None:
                inst = replacement
                folded += 1
            if dest and inst.mnemonic in {"mov.u32", "mov.b32"} and len(inst.operands) == 2:
                value = value_of(inst.operands[1])
                if value is not None:
                    constants[dest] = value
            new_insts.append(inst)
        block.instructions = new_insts
    return folded


def strength_reduce_address_induction(program: PTXProgram) -> int:
    """Replace canonical loop address recomputation with pointer induction.

    The transform is structural rather than kernel-name based.  It recognizes
    the scalar row-major pattern documented for T5 and derives both byte
    strides from the actual induction step, so renamed registers and dynamic
    dimensions remain supported.
    """
    transformed = 0
    block_by_name = {block.name: block for block in program.blocks}
    predecessors: Dict[str, List[str]] = {block.name: [] for block in program.blocks}
    for block in program.blocks:
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].append(block.name)

    synthetic_id = 0

    def new_reg(prefix: str, typ: str) -> str:
        nonlocal synthetic_id
        while True:
            name = f"%__aec_{prefix}{synthetic_id}"
            synthetic_id += 1
            if name not in program.register_types:
                program.register_types[name] = typ
                return name

    for header in program.blocks:
        if not header.instructions:
            continue
        tail = header.instructions[-1]
        if tail.mnemonic != "bra" or not tail.guard or len(header.successors) != 2:
            continue
        body_name = header.successors[1]
        body = block_by_name.get(body_name)
        if body is None or not body.instructions:
            continue
        back = body.instructions[-1]
        if back.mnemonic != "bra" or back.guard or back.operands != [header.name]:
            continue
        outside_preds = [name for name in predecessors[header.name] if name != body.name]
        if len(outside_preds) != 1:
            continue
        preheader = block_by_name[outside_preds[0]]

        seq_index = None
        fields = None
        for i in range(len(body.instructions) - 5):
            seq = body.instructions[i : i + 6]
            if not (
                seq[0].mnemonic == "mad.lo.u32"
                and seq[1].mnemonic == "mul.wide.u32"
                and seq[2].mnemonic == "add.u64"
                and seq[3].mnemonic == "mad.lo.u32"
                and seq[4].mnemonic == "mul.wide.u32"
                and seq[5].mnemonic == "add.u64"
                and seq[1].operands[1] == seq[0].operands[0]
                and seq[2].operands[2] == seq[1].operands[0]
                and seq[4].operands[1] == seq[3].operands[0]
                and seq[5].operands[2] == seq[4].operands[0]
            ):
                continue
            try:
                scale_a = parse_int(seq[1].operands[2], seq[1].line)
                scale_b = parse_int(seq[4].operands[2], seq[4].line)
            except CompileError:
                continue
            if scale_a != 4 or scale_b != 4:
                continue
            # A[row, k] and B[k, col].
            induction = seq[0].operands[3]
            if seq[3].operands[1] != induction:
                continue
            seq_index = i
            fields = (seq, induction, seq[3].operands[2], seq[2].operands[0], seq[5].operands[0])
            break
        if seq_index is None or fields is None:
            continue
        seq, induction, leading_dim_b, address_a, address_b = fields

        increment_index = None
        step = None
        for i in range(seq_index + 6, len(body.instructions) - 1):
            candidate = body.instructions[i]
            if (
                candidate.mnemonic == "add.u32"
                and len(candidate.operands) == 3
                and candidate.operands[0] == induction
                and candidate.operands[1] == induction
            ):
                increment_index, step = i, candidate.operands[2]
                break
        if increment_index is None or step is None:
            continue
        # Do not transform if k is otherwise redefined in the body.
        unsafe = False
        for i, candidate in enumerate(body.instructions):
            defs, _ = instruction_defs_uses(candidate)
            if induction in defs and i != increment_index:
                unsafe = True
                break
        if unsafe:
            continue

        address_regs = {address_a, address_b}
        # Advancing a pointer changes its post-loop value.  Only apply the
        # transform when the address temporaries are loop-local and are not
        # observed after the induction update.
        for other in program.blocks:
            if other.name in {body.name}:
                continue
            for candidate in other.instructions:
                _, uses = instruction_defs_uses(candidate)
                if uses & address_regs:
                    unsafe = True
                    break
            if unsafe:
                break
        if not unsafe:
            for candidate in body.instructions[increment_index:]:
                defs, uses = instruction_defs_uses(candidate)
                if (defs | uses) & address_regs:
                    unsafe = True
                    break
        if unsafe:
            continue

        stride_a = new_reg("stride_a", "u64")
        stride_b = new_reg("stride_b", "u64")
        step_n = new_reg("step_n", "u32")
        line = seq[0].line

        # The original six instructions calculate the initial addresses using
        # the current induction value.  Move them to the unique preheader.
        insertion = len(preheader.instructions)
        if insertion and preheader.instructions[-1].mnemonic in {"bra", "ret"}:
            insertion -= 1
        setup = [item.clone() for item in seq]
        setup.extend(
            [
                PTXInstruction("mul.wide.u32", [stride_a, step, "4"], line),
                PTXInstruction("mul.lo.u32", [step_n, step, leading_dim_b], line),
                PTXInstruction("mul.wide.u32", [stride_b, step_n, "4"], line),
            ]
        )
        preheader.instructions[insertion:insertion] = setup

        # Remove per-iteration recomputation and advance the two byte pointers
        # once all loads from the current iteration have consumed them.
        del body.instructions[seq_index : seq_index + 6]
        increment_index -= 6
        advances = [
            PTXInstruction("add.u64", [address_a, address_a, stride_a], line),
            PTXInstruction("add.u64", [address_b, address_b, stride_b], line),
        ]
        body.instructions[increment_index:increment_index] = advances
        transformed += 1
    return transformed


class Lowerer:
    def __init__(self, program: PTXProgram):
        self.program = program
        self.param_offsets = {p.name: p.offset for p in program.params}
        self.temp_id = 0
        self.pair_groups: Dict[str, Tuple[str, str]] = {}
        self.pair_hi: Dict[str, str] = {}
        for name, typ in program.register_types.items():
            if typ in {"u64", "b64"}:
                low, high = f"{name}.lo", f"{name}.hi"
                self.pair_groups[name] = (low, high)
                self.pair_hi[low] = high

    def low(self, reg: str) -> str:
        if reg in self.pair_groups:
            return self.pair_groups[reg][0]
        return reg

    def high(self, reg: str) -> str:
        if reg not in self.pair_groups:
            raise CompileError(f"'{reg}' is not a 64-bit register")
        return self.pair_groups[reg][1]

    def temp(self) -> str:
        name = f"$tmp{self.temp_id}"
        self.temp_id += 1
        return name

    def lower(self) -> MIRProgram:
        blocks: List[MIRBlock] = []
        for ptx_block in self.program.blocks:
            block = MIRBlock(ptx_block.name, successors=list(ptx_block.successors))
            constants: Dict[Tuple[int, str], str] = {}

            def ensure_reg(token: str, typ: str = "u32") -> str:
                token = token.strip()
                if is_register(token):
                    return self.low(token)
                value = parse_int(token)
                key = (value & 0xFFFFFFFF, typ)
                if key not in constants:
                    reg = self.temp()
                    constants[key] = reg
                    block.instructions.append(MIRInstruction("LOADI", dest=reg, imm=value, line=0))
                return constants[key]

            for inst in ptx_block.instructions:
                self.lower_instruction(inst, block, ensure_reg)
            blocks.append(block)
        virtuals: Set[str] = set()
        for block in blocks:
            for inst in block.instructions:
                virtuals.update(inst.gpr_uses(self.pair_hi))
                virtuals.update(inst.gpr_defs(self.pair_hi))
        return MIRProgram(self.program.kernel, blocks, self.pair_hi, self.pair_groups, len(virtuals))

    def lower_instruction(self, inst: PTXInstruction, block: MIRBlock, ensure_reg) -> None:
        m, ops, line = inst.mnemonic, inst.operands, inst.line
        arithmetic = {
            "add.u32": ("ADD", "u32", 2),
            "sub.u32": ("SUB", "u32", 2),
            "mul.lo.u32": ("MUL", "u32", 2),
            "mad.lo.u32": ("MAD", "u32", 3),
            "and.b32": ("AND", "b32", 2),
            "or.b32": ("OR", "b32", 2),
            "xor.b32": ("XOR", "b32", 2),
            "shl.b32": ("SHL", "b32", 2),
            "shr.u32": ("SHR", "u32", 2),
            "add.f32": ("ADD", "f32", 2),
            "add.rn.f32": ("ADD", "f32", 2),
            "sub.f32": ("SUB", "f32", 2),
            "sub.rn.f32": ("SUB", "f32", 2),
            "mul.f32": ("MUL", "f32", 2),
            "mul.rn.f32": ("MUL", "f32", 2),
            "mad.f32": ("MAD", "f32", 3),
            "mad.rn.f32": ("MAD", "f32", 3),
            "fma.rn.f32": ("FMA", "f32", 3),
        }
        if m in arithmetic:
            op, typ, arity = arithmetic[m]
            if len(ops) != arity + 1:
                raise CompileError(f"{m} expects {arity + 1} operands", line)
            sources = [ensure_reg(x, typ) for x in ops[1:]]
            block.instructions.append(
                MIRInstruction(op, typ, self.low(ops[0]), sources[0], sources[1], sources[2] if arity == 3 else None, line=line)
            )
            return
        if m == "mul.wide.u32":
            if len(ops) != 3:
                raise CompileError("mul.wide.u32 expects three operands", line)
            block.instructions.append(MIRInstruction("MUL", "u32", self.low(ops[0]), ensure_reg(ops[1]), ensure_reg(ops[2]), line=line))
            block.instructions.append(MIRInstruction("LOADI", dest=self.high(ops[0]), imm=0, line=line))
            return
        if m == "add.u64":
            if len(ops) != 3:
                raise CompileError("add.u64 expects three operands", line)
            block.instructions.append(MIRInstruction("ADD", "u32", self.low(ops[0]), self.low(ops[1]), self.low(ops[2]), line=line))
            block.instructions.append(MIRInstruction("LOADI", dest=self.high(ops[0]), imm=0, line=line))
            return
        if m.startswith("mov."):
            if len(ops) != 2:
                raise CompileError(f"{m} expects two operands", line)
            typ = m.split(".")[-1]
            dst, src = ops
            if src in SPECIAL_REGS:
                block.instructions.append(MIRInstruction("CPY", typ, self.low(dst), Special(SPECIAL_REGS[src]), line=line))
            elif is_register(src):
                block.instructions.append(
                    MIRInstruction("CPY", typ, self.low(dst), self.low(src), line=line, pair_def=typ in {"u64", "b64"}, pair_src=typ in {"u64", "b64"})
                )
            elif typ in {"u64", "b64"}:
                block.instructions.append(MIRInstruction("LOADI64", dest=self.low(dst), imm=parse_int(src, line), line=line, pair_def=True))
            else:
                block.instructions.append(MIRInstruction("LOADI", dest=self.low(dst), imm=parse_int(src, line), line=line))
            return
        param_match = re.fullmatch(r"ld\.param\.(u32|u64|b32|b64)", m)
        if param_match:
            typ = param_match.group(1)
            dst, param = ops[0], memory_inner(ops[1])
            offset = self.param_offsets[param]
            addr = ensure_reg(str(offset), "u32")
            if typ in {"u64", "b64"}:
                block.instructions.append(MIRInstruction("LD", "u32", self.low(dst), addr, space="pmem", line=line))
                addr_hi = ensure_reg(str(offset + 4), "u32")
                block.instructions.append(MIRInstruction("LD", "u32", self.high(dst), addr_hi, space="pmem", line=line))
            else:
                block.instructions.append(MIRInstruction("LD", typ, self.low(dst), addr, space="pmem", line=line))
            return
        global_load = re.fullmatch(r"ld\.global\.(f32|u32|b32)", m)
        if global_load:
            typ = global_load.group(1)
            block.instructions.append(MIRInstruction("LD", typ, self.low(ops[0]), self.low(memory_inner(ops[1])), space="gmem", line=line))
            return
        global_store = re.fullmatch(r"st\.global\.(f32|u32|b32)", m)
        if global_store:
            typ = global_store.group(1)
            block.instructions.append(MIRInstruction("ST", typ, src1=self.low(memory_inner(ops[0])), src2=ensure_reg(ops[1], typ), space="gmem", line=line))
            return
        compare = re.fullmatch(r"setp\.(eq|ne|lt|le|gt|ge)\.u32", m)
        if compare:
            block.instructions.append(
                MIRInstruction("CMPP", "u32", ops[0], ensure_reg(ops[1]), ensure_reg(ops[2]), subop=COMPARE_SUBOPS[compare.group(1)], line=line)
            )
            return
        if m == "bra":
            if inst.guard:
                block.instructions.append(MIRInstruction("BRX", target=ops[0], pred=inst.guard, pred_neg=inst.guard_neg, line=line))
            else:
                block.instructions.append(MIRInstruction("BR", target=ops[0], line=line))
            return
        if m == "ret":
            block.instructions.append(MIRInstruction("HALT", line=line))
            return
        raise CompileError(f"lowering not implemented for '{m}'", line)


def block_liveness(program: MIRProgram, predicate: bool = False) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    block_map = {b.name: b for b in program.blocks}
    use: Dict[str, Set[str]] = {}
    defs: Dict[str, Set[str]] = {}
    for block in program.blocks:
        b_use: Set[str] = set()
        b_def: Set[str] = set()
        for inst in block.instructions:
            iuse = inst.pred_uses() if predicate else inst.gpr_uses(program.pair_hi)
            idef = inst.pred_defs() if predicate else inst.gpr_defs(program.pair_hi)
            b_use.update(iuse - b_def)
            b_def.update(idef)
        use[block.name], defs[block.name] = b_use, b_def
    live_in = {b.name: set() for b in program.blocks}
    live_out = {b.name: set() for b in program.blocks}
    changed = True
    while changed:
        changed = False
        for block in reversed(program.blocks):
            out = set().union(*(live_in[s] for s in block.successors if s in block_map)) if block.successors else set()
            inn = use[block.name] | (out - defs[block.name])
            if out != live_out[block.name] or inn != live_in[block.name]:
                live_in[block.name], live_out[block.name] = inn, out
                changed = True
    return live_in, live_out


def mir_dce(program: MIRProgram) -> int:
    """Remove dead lowered instructions, including unused 64-bit high halves."""
    removed = 0
    changed = True
    while changed:
        changed = False
        _, gpr_out = block_liveness(program, predicate=False)
        _, pred_out = block_liveness(program, predicate=True)
        for block in program.blocks:
            live_gpr = set(gpr_out[block.name])
            live_pred = set(pred_out[block.name])
            kept: List[MIRInstruction] = []
            for inst in reversed(block.instructions):
                gdefs, guse = inst.gpr_defs(program.pair_hi), inst.gpr_uses(program.pair_hi)
                pdefs, puse = inst.pred_defs(), inst.pred_uses()
                side_effect = inst.op in {"ST", "BR", "BRX", "HALT"}
                has_live_def = bool(gdefs & live_gpr or pdefs & live_pred)
                if (gdefs or pdefs) and not side_effect and not has_live_def:
                    removed += 1
                    changed = True
                    continue
                live_gpr.difference_update(gdefs)
                live_gpr.update(guse)
                live_pred.difference_update(pdefs)
                live_pred.update(puse)
                kept.append(inst)
            block.instructions = list(reversed(kept))
    return removed


def mir_peephole(program: MIRProgram) -> int:
    """Local peephole optimisations on the lowered MIR.

    Operates within each basic block.  Returns the total number of
    instructions eliminated or simplified.
    """
    removed = 0

    for block in program.blocks:
        # Track which virtual register currently holds each LOADI value.
        # key = (value, typ), value = register name
        imm_regs: Dict[Tuple[int, str], str] = {}
        # Track the most recent definition of each register for copy-propagation.
        # key = dest_register, value = (src_register, instruction_index_in_new_list)
        last_def: Dict[str, Tuple[str, int]] = {}
        new_insts: List[MIRInstruction] = []

        for _idx, inst in enumerate(block.instructions):
            # ── substitute through known copy chains ──
            if isinstance(inst.src1, str) and inst.src1 in last_def:
                src, def_idx = last_def[inst.src1]
                # Only substitute if the source register hasn't been redefined
                # since the copy was recorded (we rebuild last_def as we go).
                if def_idx == next(
                    (i for r, (_, i) in reversed(list(last_def.items()))
                     if r == inst.src1), def_idx
                ):
                    inst.src1 = src
            if isinstance(inst.src2, str) and inst.src2 in last_def:
                src, def_idx = last_def[inst.src2]
                if def_idx == next(
                    (i for r, (_, i) in reversed(list(last_def.items()))
                     if r == inst.src2), def_idx
                ):
                    inst.src2 = src
            if isinstance(inst.src3, str) and inst.src3 in last_def:
                src, def_idx = last_def[inst.src3]
                if def_idx == next(
                    (i for r, (_, i) in reversed(list(last_def.items()))
                     if r == inst.src3), def_idx
                ):
                    inst.src3 = src

            # ── Pattern 1: LOADI dedup ──
            if inst.op == "LOADI" and inst.dest and inst.imm is not None:
                key = (inst.imm & 0xFFFFFFFF, inst.typ)
                if key in imm_regs and imm_regs[key] != inst.dest:
                    # Replace with CPY to the existing register
                    new_insts.append(MIRInstruction(
                        "CPY", inst.typ, inst.dest, imm_regs[key], line=inst.line
                    ))
                    last_def[inst.dest] = (imm_regs[key], len(new_insts) - 1)
                    removed += 1
                    continue
                imm_regs[key] = inst.dest

            # ── Pattern 2: self-copy CPY ──
            if inst.op == "CPY" and inst.dest and isinstance(inst.src1, str):
                if inst.dest == inst.src1:
                    removed += 1
                    continue  # NOP, skip entirely

            # ── Patterns 3-7: algebraic identities ──
            if inst.op in ("ADD", "SUB") and inst.typ == "u32":
                if isinstance(inst.src2, str):
                    pass  # not a constant — can't simplify
                elif (inst.op == "ADD" and inst.src2 == 0) or \
                     (inst.op == "SUB" and inst.src2 == 0):
                    if inst.dest and isinstance(inst.src1, str):
                        new_insts.append(MIRInstruction(
                            "CPY", inst.typ, inst.dest, inst.src1, line=inst.line
                        ))
                        last_def[inst.dest] = (inst.src1, len(new_insts) - 1)
                        removed += 1
                        continue
            elif inst.op == "MUL" and inst.typ == "u32":
                if isinstance(inst.src2, str):
                    pass
                elif inst.src2 == 1:
                    if inst.dest and isinstance(inst.src1, str):
                        new_insts.append(MIRInstruction(
                            "CPY", inst.typ, inst.dest, inst.src1, line=inst.line
                        ))
                        last_def[inst.dest] = (inst.src1, len(new_insts) - 1)
                        removed += 1
                        continue
                elif inst.src2 == 0:
                    if inst.dest:
                        new_insts.append(MIRInstruction(
                            "LOADI", dest=inst.dest, imm=0, line=inst.line
                        ))
                        imm_regs[(0, inst.typ)] = inst.dest
                        removed += 1
                        continue
            elif inst.op in ("SHL", "SHR") and inst.typ == "u32":
                if not isinstance(inst.src2, str) and inst.src2 == 0:
                    if inst.dest and isinstance(inst.src1, str):
                        new_insts.append(MIRInstruction(
                            "CPY", inst.typ, inst.dest, inst.src1, line=inst.line
                        ))
                        last_def[inst.dest] = (inst.src1, len(new_insts) - 1)
                        removed += 1
                        continue
            elif inst.op in ("AND", "OR", "XOR"):
                if isinstance(inst.src1, str) and isinstance(inst.src2, str) and inst.src1 == inst.src2:
                    if inst.op in ("AND", "OR"):
                        if inst.dest:
                            new_insts.append(MIRInstruction(
                                "CPY", inst.typ, inst.dest, inst.src1, line=inst.line
                            ))
                            last_def[inst.dest] = (inst.src1, len(new_insts) - 1)
                            removed += 1
                            continue

            # ── Pattern 9: LOADI followed by CPY ──
            if inst.op == "LOADI" and inst.dest and inst.imm is not None:
                pass  # already handled by imm_regs tracking above

            # ── Update tracking structures ──
            # Invalidate imm_regs entries whose register was redefined
            if inst.dest:
                # Kill last_def for this dest (new definition invalidates old copy)
                last_def.pop(inst.dest, None)
                # Also kill any last_def entries whose *source* is this dest:
                # if %r2 was a copy of %r1 and %r1 is now redefined, %r2's
                # cached source is stale.
                last_def = {
                    k: v for k, v in last_def.items()
                    if v[0] != inst.dest
                }
                # Kill any imm_regs backed by the redefined register
                imm_regs = {k: v for k, v in imm_regs.items() if v != inst.dest}
                # Track CPY definitions for copy propagation
                if inst.op == "CPY" and isinstance(inst.src1, str) and inst.dest != inst.src1:
                    last_def[inst.dest] = (inst.src1, len(new_insts))
                # Track LOADI for dedup
                if inst.op == "LOADI" and inst.imm is not None:
                    imm_regs[(inst.imm & 0xFFFFFFFF, inst.typ)] = inst.dest

            new_insts.append(inst)

        block.instructions = new_insts

    return removed


def interference_graph(program: MIRProgram, predicate: bool = False, excluded: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    excluded = excluded or set()
    _, live_out = block_liveness(program, predicate)
    graph: Dict[str, Set[str]] = {}

    def add_node(node: str) -> None:
        if node not in excluded:
            graph.setdefault(node, set())

    for block in program.blocks:
        live = set(live_out[block.name]) - excluded
        for node in live:
            add_node(node)
        for inst in reversed(block.instructions):
            uses = inst.pred_uses() if predicate else inst.gpr_uses(program.pair_hi)
            defs = inst.pred_defs() if predicate else inst.gpr_defs(program.pair_hi)
            uses -= excluded
            defs -= excluded
            for node in uses | defs:
                add_node(node)
            for defined in defs:
                for other in live:
                    if defined != other:
                        graph[defined].add(other)
                        graph[other].add(defined)
            live.difference_update(defs)
            live.update(uses)
    return graph


def greedy_color(graph: Dict[str, Set[str]], colors: Sequence[int], kind: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    saturation: Dict[str, Set[int]] = {node: set() for node in graph}
    remaining = set(graph)
    while remaining:
        node = max(remaining, key=lambda n: (len(saturation[n]), len(graph[n]), n))
        unavailable = {result[n] for n in graph[node] if n in result}
        choice = next((c for c in colors if c not in unavailable), None)
        if choice is None:
            raise CompileError(f"{kind} allocation exhausted ({len(colors)} physical registers available)")
        result[node] = choice
        remaining.remove(node)
        for neighbor in graph[node] & remaining:
            saturation[neighbor].add(choice)
    return result


def allocate_registers(program: MIRProgram) -> Tuple[Dict[str, int], Dict[str, int], int]:
    gpr_map: Dict[str, int] = {}
    used_components: Set[str] = set()
    for block in program.blocks:
        for inst in block.instructions:
            used_components.update(inst.gpr_uses(program.pair_hi))
            used_components.update(inst.gpr_defs(program.pair_hi))
    graph = interference_graph(program)

    # ── Conservative coalescing of CPY-related scalar registers ──
    # Merge non-interfering copy pairs to eliminate CPY instructions.
    # Only scalar (non-pair) copy pairs are considered.
    # Build pair-component set first so we can exclude 64-bit pairs.
    _pair_components: Set[str] = set()
    for _base, (_lo, _hi) in program.pair_groups.items():
        _pair_components.add(_lo)
        _pair_components.add(_hi)

    coalesced: Dict[str, str] = {}
    # Collect MIR-level CPY pairs
    copy_pairs: List[Tuple[str, str]] = []
    for block in program.blocks:
        for inst in block.instructions:
            if inst.op == "CPY" and inst.dest and isinstance(inst.src1, str):
                if inst.dest != inst.src1:
                    copy_pairs.append((inst.src1, inst.dest))
    # Deduplicate and apply Briggs test
    seen_pairs: Set[Tuple[str, str]] = set()
    for src, dst in copy_pairs:
        if (src, dst) in seen_pairs or (dst, src) in seen_pairs:
            continue
        seen_pairs.add((src, dst))
        if src not in graph or dst not in graph:
            continue  # one side was already merged or is unused
        # Don't coalesce across pair components
        if src in _pair_components or dst in _pair_components:
            continue
        # Must not coalesce if src and dst interfere with each other.
        # Merging interfering nodes would give them the same colour and
        # clobber a value that is still live.
        if src in graph.get(dst, set()) or dst in graph.get(src, set()):
            continue
        # Briggs test: merged node degree must be < 256 (trivially colorable)
        merged_neighbors = (graph[src] | graph[dst]) - {src, dst}
        if len(merged_neighbors) < 256:
            # Merge src → dst
            coalesced[src] = dst
            graph[dst] = merged_neighbors
            graph.pop(src, None)
            for node in merged_neighbors:
                if node in graph:
                    graph[node].discard(src)
                    graph[node].add(dst)
            # Rewrite all MIR references from src to dst
            for blk in program.blocks:
                for inst in blk.instructions:
                    if inst.dest == src:
                        inst.dest = dst
                    if inst.src1 == src:
                        inst.src1 = dst
                    if inst.src2 == src:
                        inst.src2 = dst
                    if inst.src3 == src:
                        inst.src3 = dst
        # else: not trivially colorable after merge — skip this pair

    active_pairs = {
        base: components
        for base, components in program.pair_groups.items()
        if components[0] in used_components or components[1] in used_components
    }
    component_to_pair = {component: base for base, components in active_pairs.items() for component in components}
    pair_graph: Dict[str, Set[str]] = {base: set() for base in active_pairs}
    for base, components in active_pairs.items():
        neighbors = set().union(*(graph.get(component, set()) for component in components))
        for neighbor in neighbors:
            other = component_to_pair.get(neighbor)
            if other and other != base:
                pair_graph[base].add(other)
                pair_graph[other].add(base)
    pair_bases = greedy_color(pair_graph, list(range(0, 255, 2)), "64-bit GPR pair")
    for base, physical in pair_bases.items():
        low, high = active_pairs[base]
        gpr_map[low], gpr_map[high] = physical, physical + 1

    pair_components = set(component_to_pair)
    single_nodes = sorted(set(graph) - pair_components)
    single_map: Dict[str, int] = {}
    saturation: Dict[str, Set[int]] = {node: set() for node in single_nodes}
    remaining = set(single_nodes)
    while remaining:
        node = max(remaining, key=lambda n: (len(saturation[n]), len(graph[n]), n))
        unavailable = {single_map[n] for n in graph[node] if n in single_map}
        for neighbor in graph[node]:
            pair = component_to_pair.get(neighbor)
            if pair:
                base = pair_bases[pair]
                unavailable.update({base, base + 1})
        choice = next((color for color in range(256) if color not in unavailable), None)
        if choice is None:
            candidates = [
                candidate
                for candidate in graph
                if candidate not in pair_components and not candidate.startswith("$spill")
            ]
            if not candidates:
                raise CompileError("GPR allocation exhausted and no spillable scalar value remains")
            victim = max(candidates, key=lambda candidate: (len(graph[candidate]), candidate == node))
            raise SpillNeeded(victim)
        single_map[node] = choice
        remaining.remove(node)
        for neighbor in graph[node] & remaining:
            saturation[neighbor].add(choice)
    gpr_map.update(single_map)
    pred_graph = interference_graph(program, predicate=True)
    pred_map = greedy_color(pred_graph, list(range(8)), "predicate")
    physical_count = max(gpr_map.values(), default=-1) + 1
    return gpr_map, pred_map, physical_count


def spill_virtual_register(program: MIRProgram, victim: str, slot: int, serial_start: int) -> Tuple[int, int, int]:
    """Rewrite one scalar virtual register through per-thread local memory."""
    serial = serial_start
    loads = stores = 0

    def fresh(role: str) -> str:
        nonlocal serial
        name = f"$spill_{role}{serial}"
        serial += 1
        return name

    for block in program.blocks:
        rewritten: List[MIRInstruction] = []
        for inst in block.instructions:
            uses_victim = victim in inst.gpr_uses(program.pair_hi)
            defs_victim = victim in inst.gpr_defs(program.pair_hi)
            reload_reg: Optional[str] = None
            if uses_victim:
                address = fresh("addr")
                reload_reg = fresh("load")
                rewritten.append(MIRInstruction("LOADI", dest=address, imm=slot, line=inst.line))
                rewritten.append(MIRInstruction("LD", "b32", reload_reg, address, space="lmem", line=inst.line))
                loads += 1
                if inst.src1 == victim:
                    inst.src1 = reload_reg
                if inst.src2 == victim:
                    inst.src2 = reload_reg
                if inst.src3 == victim:
                    inst.src3 = reload_reg
            stored_reg: Optional[str] = None
            if defs_victim:
                stored_reg = fresh("value")
                if inst.dest != victim or inst.pair_def:
                    raise CompileError("internal error: attempted to spill a non-scalar definition", inst.line)
                inst.dest = stored_reg
            rewritten.append(inst)
            if defs_victim:
                address = fresh("addr")
                rewritten.append(MIRInstruction("LOADI", dest=address, imm=slot, line=inst.line))
                rewritten.append(MIRInstruction("ST", "b32", src1=address, src2=stored_reg, space="lmem", line=inst.line))
                stores += 1
        block.instructions = rewritten
    return loads, stores, serial


def schedule_block(block: MIRBlock, pair_hi: Dict[str, str]) -> None:
    if len(block.instructions) < 3:
        return
    terminators = {"BR", "BRX", "HALT"}
    body = list(block.instructions)
    tail: List[MIRInstruction] = []
    while body and body[-1].op in terminators:
        tail.insert(0, body.pop())
    n = len(body)
    if n < 2:
        return
    deps: List[Set[int]] = [set() for _ in range(n)]
    succ: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        uses_i, defs_i = body[i].gpr_uses(pair_hi), body[i].gpr_defs(pair_hi)
        for j in range(i + 1, n):
            uses_j, defs_j = body[j].gpr_uses(pair_hi), body[j].gpr_defs(pair_hi)
            conflict = bool(defs_i & (uses_j | defs_j) or uses_i & defs_j)
            mem_i = body[i].op in {"LD", "ST"}
            mem_j = body[j].op in {"LD", "ST"}
            if mem_i and mem_j and body[i].space == body[j].space and (body[i].op == "ST" or body[j].op == "ST"):
                conflict = True
            if conflict:
                deps[j].add(i)
                succ[i].add(j)

    latency = []
    for inst in body:
        if inst.op == "LD":
            latency.append(600 if inst.space == "gmem" else 40)
        elif inst.op in {"MUL", "MAD", "FMA"} and inst.typ == "f32":
            latency.append(4)
        else:
            latency.append(1)
    height = [latency[i] for i in range(n)]
    for i in reversed(range(n)):
        if succ[i]:
            height[i] = latency[i] + max(height[j] for j in succ[i])
    ready = [i for i in range(n) if not deps[i]]
    emitted: List[int] = []
    done: Set[int] = set()
    while ready:
        idx = max(ready, key=lambda i: (height[i], body[i].op == "LD", -i))
        ready.remove(idx)
        emitted.append(idx)
        done.add(idx)
        for j in succ[idx]:
            if j not in done and j not in ready and deps[j] <= done:
                ready.append(j)
    if len(emitted) != n:
        raise CompileError(f"internal scheduler dependency cycle in block '{block.name}'")
    block.instructions = [body[i] for i in emitted] + tail


def encode_program(program: MIRProgram, gpr: Dict[str, int], preds: Dict[str, int]) -> Tuple[bytes, List[Dict[str, int]]]:
    # Remove self-copy CPY instructions (dest == src1 after coalescing).
    # These are NOPs and waste an instruction slot.
    removed = 0
    for block in program.blocks:
        kept: List[MIRInstruction] = []
        for inst in block.instructions:
            if inst.op == "CPY" and inst.dest and isinstance(inst.src1, str) and inst.dest == inst.src1:
                removed += 1
                continue
            kept.append(inst)
        block.instructions = kept

    pc_by_block: Dict[str, int] = {}
    pc = 0
    for block in program.blocks:
        pc_by_block[block.name] = pc
        pc += len(block.instructions)
    if pc == 0:
        raise CompileError("generated program contains no instructions")

    words: List[Tuple[int, int, int, int]] = []
    diagnostics: List[Dict[str, int]] = []

    def reg_of(value: Operand, line: int) -> int:
        if isinstance(value, Special):
            return value.selector
        if not isinstance(value, str) or value not in gpr:
            raise CompileError(f"internal error: unallocated operand '{value}'", line)
        return gpr[value]

    for block in program.blocks:
        for inst in block.instructions:
            opcode = OPCODES[inst.op]
            typ = TYPE_CODES[inst.typ if inst.op not in {"BR", "BRX", "HALT", "LOADI", "LOADI64"} else "none"]
            ctrl = (typ & 0xF) << 3
            ctrl |= (inst.subop & 0x7) << 8
            if inst.space:
                ctrl |= (SPACE_CODES[inst.space] & 0x7) << 11
            dest = src1 = src2 = immext = 0
            if inst.op == "CMPP":
                if inst.dest not in preds:
                    raise CompileError(f"unallocated predicate '{inst.dest}'", inst.line)
                dest = preds[inst.dest]
                src1, src2 = reg_of(inst.src1, inst.line), reg_of(inst.src2, inst.line)
            elif inst.op in {"ADD", "SUB", "MUL", "AND", "OR", "XOR", "SHL", "SHR"}:
                dest, src1, src2 = reg_of(inst.dest, inst.line), reg_of(inst.src1, inst.line), reg_of(inst.src2, inst.line)
            elif inst.op in {"MAD", "FMA"}:
                dest, src1, src2 = reg_of(inst.dest, inst.line), reg_of(inst.src1, inst.line), reg_of(inst.src2, inst.line)
                immext = reg_of(inst.src3, inst.line)
            elif inst.op == "LD":
                dest, src1 = reg_of(inst.dest, inst.line), reg_of(inst.src1, inst.line)
            elif inst.op == "ST":
                src1, src2 = reg_of(inst.src1, inst.line), reg_of(inst.src2, inst.line)
            elif inst.op == "CPY":
                dest, src1 = reg_of(inst.dest, inst.line), reg_of(inst.src1, inst.line)
            elif inst.op == "LOADI":
                dest, immext = reg_of(inst.dest, inst.line), int(inst.imm or 0) & 0xFFFFFFFF
            elif inst.op == "LOADI64":
                dest = reg_of(inst.dest, inst.line)
                value = int(inst.imm or 0) & 0xFFFFFFFFFFFFFFFF
                src2, immext = (value >> 32) & 0xFFFFFFFF, value & 0xFFFFFFFF
                if dest > 254:
                    raise CompileError("LOADI64 destination pair exceeds R255", inst.line)
            elif inst.op == "BR":
                if inst.target not in pc_by_block:
                    raise CompileError(f"unknown branch target '{inst.target}'", inst.line)
                immext = pc_by_block[inst.target]
            elif inst.op == "BRX":
                if inst.target not in pc_by_block or inst.pred not in preds:
                    raise CompileError("invalid conditional branch target or predicate", inst.line)
                ctrl |= preds[inst.pred] & 0x7
                ctrl |= 1 << 15
                if inst.pred_neg:
                    ctrl |= 1 << 14
                immext = pc_by_block[inst.target]
            elif inst.op != "HALT":
                raise CompileError(f"cannot encode '{inst.op}'", inst.line)

            if not (0 <= dest <= 0xFFFF and 0 <= src1 <= 0xFFFF and 0 <= src2 <= 0xFFFFFFFF):
                raise CompileError("encoded operand field out of range", inst.line)
            w0 = immext & 0xFFFFFFFF
            w1 = src2 & 0xFFFFFFFF
            w2 = ((dest & 0xFFFF) << 16) | (src1 & 0xFFFF)
            w3 = ((opcode & 0xFFFF) << 16) | (ctrl & 0xFFFF)
            words.append((w0, w1, w2, w3))
            diagnostics.append({"opcode": opcode, "ctrl": ctrl, "dest": dest, "src1": src1, "src2": src2, "immext": w0})
    return b"".join(struct.pack("<IIII", *word) for word in words), diagnostics


def validate_binary(data: bytes, diagnostics: List[Dict[str, int]]) -> None:
    if not data or len(data) % 16:
        raise CompileError("generated .aecbin is empty or not 16-byte aligned")
    instruction_count = len(data) // 16
    valid_opcodes = set(OPCODES.values())
    for pc in range(instruction_count):
        w0, w1, w2, w3 = struct.unpack_from("<IIII", data, pc * 16)
        opcode, ctrl = w3 >> 16, w3 & 0xFFFF
        if opcode not in valid_opcodes:
            raise CompileError(f"internal validation found invalid opcode at PC {pc}")
        typ = (ctrl >> 3) & 0xF
        if typ not in set(TYPE_CODES.values()):
            raise CompileError(f"internal validation found invalid type at PC {pc}")
        if opcode in {OPCODES["BR"], OPCODES["BRX"]} and w0 >= instruction_count:
            raise CompileError(f"branch target {w0} at PC {pc} is outside the program")
        if opcode == OPCODES["LOADI64"] and (w2 >> 16) > 254:
            raise CompileError(f"LOADI64 pair at PC {pc} exceeds R255")
    if len(diagnostics) != instruction_count:
        raise CompileError("internal encoder diagnostic length mismatch")


def count_stats(program: MIRProgram) -> Dict[str, int]:
    stats = {"branch_count": 0, "load_count": 0, "store_count": 0}
    for block in program.blocks:
        for inst in block.instructions:
            if inst.op in {"BR", "BRX"}:
                stats["branch_count"] += 1
            elif inst.op == "LD":
                stats["load_count"] += 1
            elif inst.op == "ST":
                stats["store_count"] += 1
    return stats


def compile_source(source: str, input_name: str, output_name: str, opt_level: str) -> Tuple[bytes, Dict[str, object]]:
    ptx = parse_ptx(source)
    pass_stats = {
        "block_merging": 0,
        "constant_folding": 0,
        "cse": 0,
        "dce": 0,
        "licm": 0,
        "mir_dce": 0,
        "mir_peephole": 0,
        "mad_fusion": 0,
        "address_induction": 0,
    }
    if opt_level == "O2":
        pass_stats["block_merging"] = merge_blocks(ptx)
        pass_stats["licm"] = licm(ptx)
        pass_stats["address_induction"] = strength_reduce_address_induction(ptx)

        # Fixpoint: fold constants and eliminate dead code iteratively.
        # Each round may expose new constant operands or dead instructions
        # that enable further folding in the next round.
        while True:
            fc = fold_integer_constants(ptx)
            dc = dce(ptx)
            pass_stats["constant_folding"] += fc
            pass_stats["dce"] += dc
            if fc == 0 and dc == 0:
                break

        pass_stats["cse"] = local_cse(ptx)
        pass_stats["dce"] += dce(ptx)       # CSE may create dead copies
        pass_stats["mad_fusion"] = fuse_mad(ptx)
        pass_stats["dce"] += dce(ptx)       # MAD fusion may make old mul/add dead
    lowerer = Lowerer(ptx)
    mir = lowerer.lower()
    if opt_level == "O2":
        pass_stats["mir_dce"] = mir_dce(mir)
        pass_stats["mir_peephole"] = mir_peephole(mir)
        pass_stats["mir_dce"] += mir_dce(mir)  # peephole may create dead code
        for block in mir.blocks:
            schedule_block(block, mir.pair_hi)
    spill_loads = spill_stores = spill_serial = 0
    spill_slot = 0
    while True:
        try:
            gpr_map, pred_map, physical_count = allocate_registers(mir)
            break
        except SpillNeeded as request:
            loads, stores, spill_serial = spill_virtual_register(mir, request.virtual_register, spill_slot, spill_serial)
            spill_loads += loads
            spill_stores += stores
            spill_slot += 4
            if spill_slot > 1024 * 1024:
                raise CompileError("local-memory spill area exceeded 1 MiB per thread")
    binary, diagnostics = encode_program(mir, gpr_map, pred_map)
    validate_binary(binary, diagnostics)
    stats = count_stats(mir)
    num_instructions = len(binary) // 16
    report: Dict[str, object] = {
        "status": "ok",
        "input": input_name,
        "output": output_name,
        "kernel": ptx.kernel,
        "opt_level": opt_level,
        "num_ptx_instructions": ptx.num_source_instructions,
        "num_aec_instructions": num_instructions,
        "num_basic_blocks": len(ptx.blocks),
        "num_virtual_registers": mir.num_virtual_registers,
        "num_physical_registers": physical_count,
        "num_predicates": len(set(pred_map.values())),
        "spills": {"loads": spill_loads, "stores": spill_stores},
        "passes": {
            "block_merging": opt_level == "O2",
            "dce": opt_level == "O2",
            "cse": opt_level == "O2",
            "licm": opt_level == "O2",
            "mad_fusion": opt_level == "O2",
            "address_induction": opt_level == "O2",
            "scheduler": "latency-list" if opt_level == "O2" else "source-order",
        },
        "pass_statistics": pass_stats,
        **stats,
        "memory_instruction_ratio": (stats["load_count"] + stats["store_count"]) / num_instructions,
        "warnings": [],
    }
    return binary, report


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aec-cc", description="Compile the C1 PTX subset to AEC machine code")
    parser.add_argument("input", help="input PTX file")
    parser.add_argument("-o", "--output", required=True, help="output .aecbin file")
    optimization = parser.add_mutually_exclusive_group()
    optimization.add_argument("-O0", dest="opt_level", action="store_const", const="O0")
    optimization.add_argument("-O2", dest="opt_level", action="store_const", const="O2")
    optimization.add_argument("-O3", dest="opt_level", action="store_const", const="O2")
    parser.set_defaults(opt_level="O0")
    parser.add_argument("--report", help="write compile report JSON")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else None
    try:
        source = input_path.read_text(encoding="utf-8")
        binary, report = compile_source(source, str(input_path), str(output_path), args.opt_level)
        atomic_write(output_path, binary)
        if report_path:
            atomic_write(report_path, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        return 0
    except (OSError, CompileError) as exc:
        error = str(exc)
        if report_path:
            failure = {
                "status": "error",
                "input": str(input_path),
                "output": str(output_path),
                "opt_level": args.opt_level,
                "error": error,
            }
            try:
                atomic_write(report_path, (json.dumps(failure, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError:
                pass
        print(f"aec-cc: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
