from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import os
from typing import List, Dict
import re

# ============================================================
# FIXED: Corrected "FASTAPI" to "FastAPI" and "tile" to "title"
# ============================================================
app = FastAPI(
    title="RAG Resume Screening API",
    description="API for intelligent resume screening using RAG + LLM",
    version="1.0"
)

# ============================================================
# ADDED: CORS Middleware for Hugging Face compatibility
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# FIXED: Added quotes around filename
# ============================================================
with open('rag_system.pkl', "rb") as f:
    rag_data = pickle.load(f)

embedding_model = rag_data["embedding_model"]
vector_index = rag_data["vector_index"]
chunks = rag_data["chunks"]
metadata = rag_data["metadata"]

# Load LLM
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Text preprocessing
# ============================================================
def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s\.\,\-\:\(\)]", "", text)
    return text

def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        import PyPDF2
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith(".docx"):
        import docx  # FIXED: changed "docs" to "docx"
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()

# ============================================================
# FIXED: "strt" to "str"
# ============================================================
class JobDescription(BaseModel):
    text: str
    top_k: int = 5

# ============================================================
# FIXED: Multiple syntax errors in retrieve_chunks
# ============================================================
def retrieve_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query])[0].astype("float32")  # FIXED: missing closing bracket
    distances, indices = vector_index.search(query_embedding.reshape(1, -1), top_k)  # FIXED: "seaerch" -> "search"
    
    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append({  # FIXED: "retrieve_chunks" -> "retrieved_chunks"
                "chunk": chunks[idx][:500],  # FIXED: "chunks" -> "chunk"
                "metadata": metadata[idx],
                "distance": float(distances[0][list(indices[0]).index(idx)])  # FIXED: missing closing parentheses
            })
            
    return retrieved_chunks  # FIXED: "retrieve_chunks" -> "retrieved_chunks"

# ============================================================
# FIXED: generate_evaluation function
# ============================================================
def generate_evaluation(job_description, resume_text):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(job_description, top_k=5)
    
    # Construct prompt - FIXED: missing closing braces
    prompt = f"""
    Job Description:
    {job_description[:1000]}
    
    Candidate Resume:
    {resume_text[:1500]}
    
    Based on the job description and resume, provide a structured evaluation:
    1. Overall Match Score: [0-100]
    2. Key Strengths: [list 3-5 items]
    3. Key Weaknesses: [list 2-3 items]
    4. Skills Matched: [list]
    5. Skills Missing: [list]
    6. Recommendation: [Hire/Interview/Reject]
    
    Evaluation:
    """
    
    # Generate response - FIXED: "input" -> "inputs"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1000, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,  # FIXED: "input" -> "inputs"
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
            
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the evaluation
    try:
        evaluation_text = response.split("Evaluation:")[1].strip()  # FIXED: added colon
    except:
        evaluation_text = response
        
    # Simple parsing
    score = 50
    strengths = []
    weaknesses = []
    matched_skills = []
    missing_skills = []
    recommendation = "Review"
    
    lines = evaluation_text.split("\n")  # FIXED: missing dot
    for line in lines:
        line_lower = line.lower()
        if "score" in line_lower and ("%" in line or "0" in line):
            try:
                score_match = re.search(r"(\d+)", line)
                if score_match:
                    score = int(score_match.group(1))  # FIXED: missing closing parenthesis
                    if score > 100:
                        score = 50
            except:
                pass
        elif "strength" in line_lower:
            strengths.append(line.strip())
        elif "weakness" in line_lower:  # FIXED: "weaknesses" -> "weakness"
            weaknesses.append(line.strip())
        elif "skill" in line_lower and "match" in line_lower:
            matched_skills.append(line.strip())
        elif "missing" in line_lower:
            missing_skills.append(line.strip())
        elif "recommendation" in line_lower:
            recommendation = line.split(":")[-1].strip()
            
    # if no strengths found, add some default ones
    if not strengths:
        skills_found = re.findall(r"([A-Z][a-z]+[A-Za-z]*)", resume_text[:500])  # FIXED: missing closing bracket
        if skills_found:
            strengths = [f"Candidate has skills in {', '.join(skills_found[:3])}"]  # FIXED: quotes
        else:
            strengths = ["Candidate has relevant experience", "Good technical background"]  # FIXED: spelling
    
    if not weaknesses:
        weaknesses = ["Limited information available", "Review in detail"]  # FIXED: spelling
        
    if not matched_skills:
        matched_skills = ["Skills identified in resume"]  # FIXED: spelling
        
    if not missing_skills:
        missing_skills = ["Complete skill assessment needed"]  # FIXED: braces to brackets
        
    return {
        "score": score,
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:3],
        "skills_matched": matched_skills[:5],
        "missing_skills": missing_skills[:5],
        "recommendation": recommendation if recommendation else "Review",
        "detailed_evaluation": evaluation_text
    }

# ============================================================
# FIXED: read_root - missing closing bracket
# ============================================================
@app.get("/")
def read_root():
    return {"message": "RAG Resume Screening API - Powered by AI"}  # FIXED: braces

# ============================================================
# FIXED: screen_candidate endpoint
# ============================================================
@app.post("/screen_candidate")
async def screen_candidate(
    job_description: str,
    resume_file: UploadFile = File(...)  # FIXED: lowercase "file"
):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.filename)[1]) as tmp_file:  # FIXED: "splittext" -> "splitext"
            content = await resume_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
                
        # Extract text
        resume_text = extract_text_from_file(tmp_file_path)
        resume_text = preprocess_text(resume_text)  # FIXED: missing closing parenthesis
            
        # Get evaluation
        evaluation = generate_evaluation(job_description, resume_text)
            
        # Get relevant chunks
        relevant_chunks = retrieve_chunks(job_description, top_k=3)
            
        os.unlink(tmp_file_path)
            
        return {
            "filename": resume_file.filename,
            "match_score": evaluation["score"],
            "strengths": evaluation["strengths"],  # FIXED: "strength" -> "strengths"
            "weaknesses": evaluation["weaknesses"],  # FIXED: "weeaknesses" -> "weaknesses"
            "skills_matched": evaluation["skills_matched"],
            "missing_skills": evaluation["missing_skills"],
            "recommendation": evaluation["recommendation"],
            "detailed_evaluation": evaluation["detailed_evaluation"],
            "relevant_experience": [chunk["chunk"] for chunk in relevant_chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing candidate: {str(e)}")

# ============================================================
# FIXED: compare_candidates endpoint
# ============================================================
@app.post("/compare_candidates")  # FIXED: plural
async def compare_candidates(  # FIXED: plural
    job_description: str,
    resume_files: List[UploadFile] = File(...)
):
    try:
        results = []  # FIXED: missing equals
        
        for file in resume_files:  # FIXED: "resume_file" -> "resume_files"
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:  # FIXED: missing [1]
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
                    
            try:
                resume_text = extract_text_from_file(tmp_file_path)
                resume_text = preprocess_text(resume_text)
                        
                evaluation = generate_evaluation(job_description, resume_text)
                        
                results.append({
                    "filename": file.filename,
                    "score": evaluation["score"],
                    "recommendation": evaluation["recommendation"],
                    "strengths": evaluation["strengths"],
                    "weaknesses": evaluation["weaknesses"]
                })
                        
            finally:
                os.unlink(tmp_file_path)
                        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
            
        return {
            "job_description": job_description,
            "candidates": results,  # FIXED: "candidate" -> "candidates"
            "total_candidates": len(results),
            "top_candidate": results[0]["filename"] if results else None
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing candidates: {str(e)}")  # FIXED: missing closing parenthesis

# ============================================================
# FIXED: search_candidates endpoint
# ============================================================
@app.get("/search_candidates")
def search_candidates(query: str, top_k: int = 10):
    try:
        relevant_chunks = retrieve_chunks(query, top_k)
        
        # Group by resume ID
        resume_matches = {}
        for chunk in relevant_chunks:
            resume_id = chunk["metadata"]["resume_id"]
            if resume_id not in resume_matches:
                resume_matches[resume_id] = {
                    "resume_id": resume_id,
                    "name": chunk["metadata"].get("name", "Unknown"),
                    "experience_years": chunk["metadata"].get("experience_years", 0),  # FIXED: key name
                    "applied_job_role": chunk["metadata"].get("applied_job_role", "Unknown"),  # FIXED: missing quotes
                    "skills": chunk["metadata"].get("skills", ""),  # FIXED: missing quotes
                    "education": chunk["metadata"].get("education", ""),
                    "chunks": []
                }
                
            resume_matches[resume_id]["chunks"].append(chunk["chunk"])
                
        return {
            "query": query,
            "num_results": len(resume_matches),
            "results": list(resume_matches.values())  # FIXED: missing closing parenthesis
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")  # FIXED: missing closing parenthesis

# ============================================================
# FIXED: candidate_details endpoint
# ============================================================
@app.get("/candidate_details")
def get_candidate_details(candidate_id: int):
    try:
        if candidate_id < len(metadata):
            # Find all chunks for this candidate
            candidate_chunks = []
            for i, meta in enumerate(metadata):
                if meta["resume_id"] == candidate_id:
                    candidate_chunks.append({
                        "chunk": chunks[i],
                        "metadata": meta
                    })  # FIXED: missing closing parenthesis
              
            if candidate_chunks:
                return {
                    "candidate_id": candidate_id,
                    "name": candidate_chunks[0]["metadata"]["name"],
                    "experience_years": candidate_chunks[0]["metadata"]["experience_years"],
                    "skills": candidate_chunks[0]["metadata"]["skills"],  # FIXED: was "education"
                    "education": candidate_chunks[0]["metadata"]["education"],  # FIXED: was missing
                    "applied_job_role": candidate_chunks[0]["metadata"]["applied_job_role"],
                    "num_chunks": len(candidate_chunks),
                    "full_text": candidate_chunks[0]["metadata"].get("full_text", "Not available")
                }
                
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")  # FIXED: braces and quotes
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching candidate details: {str(e)}")  # FIXED: parentheses

# ============================================================
# FIXED: model_info endpoint
# ============================================================
@app.get("/model_info")
def get_model_info():
    return {
        "retrieval_model": "all-MiniLM-L6-v2",
        "llm_model": "microsoft/DialoGPT-large",  # FIXED: consistent name
        "embedding_dimension": 384,
        "chunks_total": len(chunks),
        "device": device
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
