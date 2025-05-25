from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from models.dna_model import DNALanguageModel
from models.protein_model import ProteinLanguageModel

app = FastAPI(
    title="LLM Research Tool API",
    description="API for DNA/Protein sequence analysis and HLA variant prediction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SequenceRequest(BaseModel):
    sequence: str

class BatchSequenceRequest(BaseModel):
    sequences: List[str]

class PredictionResponse(BaseModel):
    sequence: str
    predictions: dict
    error: Optional[str] = None

# Dependency to get model instances
def get_dna_model():
    try:
        return DNALanguageModel()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_protein_model():
    try:
        return ProteinLanguageModel()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to LLM Research Tool API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# DNA Model Endpoints
@app.post("/dna/predict", response_model=PredictionResponse)
async def predict_dna(
    request: SequenceRequest,
    model: DNALanguageModel = Depends(get_dna_model)
):
    try:
        predictions = model.predict(request.sequence)
        return {
            "sequence": request.sequence,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dna/batch-predict", response_model=List[PredictionResponse])
async def batch_predict_dna(
    request: BatchSequenceRequest,
    model: DNALanguageModel = Depends(get_dna_model)
):
    try:
        results = model.batch_predict(request.sequences)
        return [
            {
                "sequence": seq,
                "predictions": pred if "error" not in pred else None,
                "error": pred.get("error")
            }
            for seq, pred in zip(request.sequences, results)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Protein Model Endpoints
@app.post("/protein/predict", response_model=PredictionResponse)
async def predict_protein(
    request: SequenceRequest,
    model: ProteinLanguageModel = Depends(get_protein_model)
):
    try:
        predictions = model.predict(request.sequence)
        return {
            "sequence": request.sequence,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/protein/batch-predict", response_model=List[PredictionResponse])
async def batch_predict_protein(
    request: BatchSequenceRequest,
    model: ProteinLanguageModel = Depends(get_protein_model)
):
    try:
        results = model.batch_predict(request.sequences)
        return [
            {
                "sequence": seq,
                "predictions": pred if "error" not in pred else None,
                "error": pred.get("error")
            }
            for seq, pred in zip(request.sequences, results)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/protein/embeddings")
async def get_protein_embeddings(
    sequence: str,
    model: ProteinLanguageModel = Depends(get_protein_model)
):
    try:
        embeddings = model.get_embeddings(sequence)
        return {"sequence": sequence, "embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 