from fastapi import FASTAPI, UploadFile, File, HTTPException
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

app = FastAPI(
    tile="RAG Resume Screening API",
    description="API for intelligent resume screening using RAG + LLM",
    version="1.0
    )
    
# Load RAG components
with open(rag_system.pkl, "rb") as f:
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

# Text preprocessing
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
        import docs
        doc = docx.Document(file_path) # loads the word document
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
            
class JobDescription(BaseModel):
    text: strt
    top_k: int = 5
    
def retrieve_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query][0].astype("float32")
    distances, indices = vector_index.seaerch(query_embedding.reshape(1, -1), top_k)
    
    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieve_chunks.append({
                "chunks": chunks[idx][:500],
                "metadata": metadata[idx],
                "distance": float(distance[0][list(indices[0].index(idx)
            })
            
    return retrieve_chunks
    
def generate_evaluation(job_description, resume_text):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(job_description, top_k=5)
    
    # Construct prompt
    prompt = f"""
    Job Description:
    {job_description[:100]}
    
    Candidate Resume:
    {resume_text[:1500]
    
    Based on the job description and resume, provide a structured evaluation:
    1. Overall Match Score: [0, 100]
    2. Key Strengths: [list 3-5 items]
    3. Key Weaknesses: [list 2-3 items]
    4. Skills Matched: [list]
    5. Skills Missing: [list]
    6. Recommendation: [Hire/Interview/Reject]
    
    Evaluation
    """
    
    # Genereate response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1000, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(
            **input,
            max_new_tokens=400
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
            )
            
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #Parse the evaluation
    try:
        evaluation_text = response.split("Evaluation")[1].strip()
    except:
        evaluation_text = response
        
    # Simple parsing
    score = 50
    strengths = []
    weaknesses = []
    matched_skills = []
    missing_skills = []
    recommendation = "Review"
    
    lines = evaluation_text_split("\n")
    for line in lines:
        line_lower = line.lower()
        if "score" in line_lower and ("%" in line or "0" in line):
            try:
                score_match = re.search(r"(\d+)", line)
                if score_match:
                    score = int(score_match.group(1)
                    if score > 100:
                        score = 50
                        
            except:
                pass
        elif "strength" in line_lower:
            strengths.append(line.strip())
        elif "weaknesses" in line_lower:
            weaknesses.append(line.strip())
        elif "skill" in line_lower and "match" in line_lower:
            matched_skills.append(line.strip())
        elif "missing" in line_lower:
            missing_skills.append(line.strip())
        elif "recommendation" in line_lower:
            recommendation = line.split(":")[-1].strip()
            
    # if no strengths found, add some default ones
    if not strengths:
        # Try to extract skills from resume
        skills_found = re.findall(r"([A-Z][a-z]+[A-Za-Z]*)", resume_text[:500]
        if skills_found:
            strengths = [f"Candidate has skills in {", ".join(skills_found[:3])}]
        else:
            strengths = ["Candidate has relevant experience", "Good technical back ground"]
    
    if not weaknesses:
        weaknesses = ["Limited infromation available", "Review in detail"]
        
        if not matched_skills:
            matched_skills = ["Skills indentified in resume"]
            
        if not missing_skills:
            missing_skills = ["Complete skill assessment needed"}
            
        return {
            "score": score,
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:3],
            "skills_matched": matched_skills[:5],
            "missing_skills": missing_skills[:5],
            "recommendation": recommendation if recommendation else "Review",
            "detailed_evaluation": evaluation_text
        }
        
@app.get("/")
def read_root():
    return{"message": "RAG Resume Screening API - Powered by AI")
    
@app.post("/screen_candidate")
async def screen_candidate(
            job_description: str,
            resume_file: UploadFile = file(...)
    ):
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splittext(resume_file.filename)[1]) as tmp_file:
                content = await resume_file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
                
            # Extract text
            resume_text = extract_text_from_file(tmp_file_path)
            resume_text = preprocess_text(resume_text
            
            # Get evaluation
            evaluation = generate_evaluation(job_description, resume_text)
            
            # Get relevant chunks
            relevant_chunks = retrieve_chunks(job_description, top_k=3)
            
            os.unlink(tmp_file_path)
            
            return {
                "filename": resume_file.filename,
                "match_score": evaluation["score"],
                "strengths": evaluation["strength"],
                "weeaknesses": evaluation["weaknesses"],
                "skills_matched": evaluation["skills_matched"],
                "missing_skills": evaluation["missing_skills"],
                "recommendation": evaluation["recommendation"],
                "detailed_evaluation": evaluation["detailed_evaluation"],
                "relevant_experience": [chunk["chunk"] for chunk in relevant_chunks]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing candidate: {str(e)}")
    
@app.post("/compare_candidate")
async def compare_candidate(
        job_description: str,
        resume_files: List[UploadFile] = File(...)
    ):
        try:
            results []
            
            for file in resume_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)) as tmp_file:
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
                "candidate": results,
                "total_candidates": len(results),
                "top_candidate": results[0]["filename"] if results else None
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error compare_candidate: {str(e)")
            
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
                    "experience": chunk["metadata"].get("experience_years", 0),
                    "applied_job_role: chunk["metadata"].get("applied_job_role", "Unknown"),
                    "skills": chunk["metadata"].get(skills", ""),
                    "education": chunk["metadata"].get("education", ""),
                    "chunks": []
                }
                
            resume_matches[resume_id]["chunks"].append(chunk["chunk"])
                
        return {
            "query": query,
            "num_results": len(resume_matches),
            "results": list(resume_matches.values()
        }   
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)})

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
                    }
              
            if candidate_chunks:
                return {
                    "candidate_id": candidate_id,
                    "name": candidate_chunks[0]["metadata"]["name"],
                    "experience_years": candidate_chunks[0]["metadata"]["experience_years"],
                    "skills": candidate_chunks[0]["metadata"]["education"],
                    "applied_job_role": candidate_chunks[0]["metadata"]["applied_job_role"],
                    "num_chunks": len(candidate_chunks),
                    "full_text": candidate_chunks[0]["metadata"]["full_text"]
                }
                
        raise HTTPException(status_code=500, detail=f"Candidate" {candidate_id} not found")
                    
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fectching candidate details: {(e)}")

@app.get("/model_info")
def get_model_info():
    return {
        "retrieval_model": "all-MiniLM-L6-v2",
        "llm_model": "microsoft/DialoGPT-Medium",
        "embedding_dimension": 384,
        "chunks_total": len(chunks),
        "device": device
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
            
        

    
