import sys
import json
import traceback


def main():
    print("READY", flush=True)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            task = json.loads(line)
            
            if task.get("cmd") == "exit":
                sys.exit(0)
            
            onnx_path = task["onnx"]
            input_dir = task["input"]
            output_dir = task["output"]
            batch_size = task.get("batch_size")
            
            from .executor import InferenceExecutor
            
            executor = InferenceExecutor(onnx_path, batch_size)
            
            if batch_size and batch_size > 0:
                samples = executor.batch_run(input_dir, output_dir)
            else:
                samples = executor.run(input_dir, output_dir)
            
            result = {"status": "ok", "samples": int(samples)}
            print(json.dumps(result), flush=True)
            
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}), flush=True)
            sys.stderr.write(f"Error: {traceback.format_exc()}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
