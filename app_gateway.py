from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import os, shutil, uuid
import vital_engine

app = FastAPI(title="Thermal Vital Summary API")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/analyze/")
async def analyze_vitals(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trigger worker
    background_tasks.add_task(vital_engine.run_headless, task_id, input_path, OUTPUT_DIR)

    return {
        "message": "Analysis started",
        "task_id": task_id,
        "status_endpoint": f"/status/{task_id}"
    }

@app.get("/status/{task_id}")
async def get_summary(task_id: str):
    """Returns the JSON summary of the vitals."""
    result = vital_engine.task_results.get(task_id)
    if not result:
        return {"status": "Processing", "task_id": task_id}
    return result

@app.get("/download/{filename}")
async def get_plot(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Plot not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)