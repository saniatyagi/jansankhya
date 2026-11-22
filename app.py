import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="JanSahayak Civic RAG Backend")

# Allow frontend (including GitHub Pages) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grounded Knowledge Base (RAG Chunks)
RAG_CORPUS = {
    "ll": {
        "topic": "Learner's Licence (LL) Documents & Steps",
        "portal": "Sarathi Parivahan (MoRTH) — https://sarathi.parivahan.gov.in/",
        "citation": "[Source: MoRTH / Central Motor Vehicles Rules & Sarathi Parivahan Guidelines]",
        "docs": [
            "Proof of Age (Birth Certificate / Passport / 10th Marksheet)",
            "Proof of Address (Aadhaar / Voter ID / Utility Bill / Passport)",
            "Passport-size photograph & digital signature",
            "Form 1 (Physical Fitness Self-Declaration) / Form 1A (Medical Certificate if applicable)",
            "Application fee payment receipt & LL test slot confirmation",
        ],
        "context": (
            "1. Visit Sarathi Parivahan and select your State/UT. "
            "2. Apply for New Learner's Licence (LL), fill Form 2 details, and upload age, address, and photo/signature documents. "
            "3. Pay the prescribed fee and book a slot for the online/RTO Learner's Licence test. "
            "4. Pass the test to download your Learner's Licence (valid for 6 months). "
            "Caveat: Online test availability and RTO slot rules vary by State/UT."
        ),
    },
    "nsp": {
        "topic": "NSP Post-Matric Scholarship Eligibility",
        "portal": "National Scholarship Portal (NSP) — https://scholarships.gov.in/",
        "citation": "[Source: National Scholarship Portal (scholarships.gov.in) Scheme Guidelines]",
        "docs": [
            "One-Time Registration (OTR) reference / Aadhaar-linked verification",
            "Previous academic marksheet & current institution bonafide proof",
            "Income certificate issued by competent authority",
            "Category / Caste / Domicile certificate (as applicable)",
            "Active Aadhaar-seeded bank account for DBT",
        ],
        "context": (
            "1. Complete One-Time Registration (OTR) on NSP. "
            "2. Log in during the active scheme window and verify scheme-specific income, course, and marks criteria. "
            "3. Submit the Post-Matric application with supporting certificates for Institute and Nodal verification. "
            "Caveat: Income limits and cut-off dates differ across Central and State schemes."
        ),
    },
    "aadhaar": {
        "topic": "Aadhaar Address Update Procedure",
        "portal": "UIDAI myAadhaar — https://myaadhaar.uidai.gov.in/",
        "citation": "",
        "docs": [
            "Valid Proof of Address (PoA) document in applicant's name",
            "Aadhaar number & registered mobile number for OTP verification",
            "Applicable update fee receipt",
        ],
        "context": (
            "1. Log in to myAadhaar using OTP on your registered mobile number. "
            "2. Choose 'Update Aadhaar Online' -> 'Address', enter the new address, and upload a valid PoA document. "
            "3. Pay the update fee and save the Service Request Number (SRN) to track status. "
            "Caveat: If mobile number is not linked, visit an Aadhaar Seva Kendra."
        ),
    },
    "expired_dl": {
        "topic": "Conflicting Rule Test: Expired Driving Licence Renewal",
        "portal": "Sarathi Parivahan (MoRTH) — https://sarathi.parivahan.gov.in/",
        "citation": "",
        "docs": [
            "Existing / expired Driving Licence details",
            "Proof of age and address",
            "Form 1 / Form 1A (Medical certificate if age 40+ or transport category)",
            "Renewal & late fee receipt",
        ],
        "context": (
            "1. Check exact expiry date and grace period on Sarathi Parivahan. "
            "2. Submit DL renewal application online with declarations/medical form. "
            "CONFLICTING RULE FLAG: Grace periods, late penalties, and re-test rules after prolonged expiry vary between statutory amendments and State RTO enforcement."
        ),
    },
}


class ChatRequest(BaseModel):
    query: str
    variant: str = "V3"
    jurisdiction: str = "All India / Central"


def retrieve_chunk(query: str):
    q = query.lower()
    if any(k in q for k in ["learner", "ll", "licence", "license"]):
        return "ll", RAG_CORPUS["ll"]
    if any(k in q for k in ["nsp", "scholarship", "post-matric"]):
        return "nsp", RAG_CORPUS["nsp"]
    if any(k in q for k in ["aadhaar", "address"]):
        return "aadhaar", RAG_CORPUS["aadhaar"]
    if any(k in q for k in ["expired", "conflict", "renewal"]):
        return "expired_dl", RAG_CORPUS["expired_dl"]
    return "general", None


def generate_llm_or_grounded_reply(query: str, variant: str, jurisdiction: str, chunk: dict):
    system_prompt = (
        f"You are JanSahayak, an Indian civic procedure assistant. "
        f"Prompt Variant: {variant}. Jurisdiction: {jurisdiction}. "
        f"Rules: Ground answers strictly in retrieved official context, cite sources in V3, and flag state-level rule variations."
    )

    # Optional: Live Gemini API call if GEMINI_API_KEY is set in environment
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and chunk:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"{system_prompt}\n\nRetrieved Context:\n{chunk['context']}\n\nUser Question: {query}"
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text, system_prompt
        except Exception:
            pass  # Fall back cleanly to deterministic RAG output

    if not chunk:
        return (
            f"I can assist with Indian government procedures across Education & Scholarships (NSP), "
            f"Identity Documents (Aadhaar), and Driving Licences (Sarathi Parivahan) for {jurisdiction}. "
            f"Ask about one of these services to retrieve official steps and documents.",
            system_prompt,
        )

    if variant == "V1":
        return f"{chunk['context']}\n\nPortal: {chunk['portal']}", system_prompt
    if variant == "V2":
        docs_str = "\n".join(f"• {d}" for d in chunk["docs"])
        return f"Procedure ({jurisdiction}):\n{chunk['context']}\n\nRequired Documents:\n{docs_str}\n\nPortal: {chunk['portal']}", system_prompt

    # V3: Evidence-Grounded & State-Aware
    docs_str = "\n".join(f"• {d}" for d in chunk["docs"])
    reply = (
        f"Verified Procedure ({jurisdiction}):\n{chunk['context']}\n\n"
        f"Required Documents:\n{docs_str}\n\n"
        f"{chunk['citation']}\nOfficial Portal: {chunk['portal']}"
    )
    return reply, system_prompt


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    # 1. Server-side PII Guardrail
    if re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", req.query):
        masked = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "XXXX-XXXX-XXXX", req.query)
        return {
            "type": "guardrail",
            "title": "Backend Privacy Guardrail Triggered",
            "body": (
                f"Server blocked and masked sensitive 12-digit ID number: \"{masked}\".\n"
                "Never submit Aadhaar numbers or OTPs in chat. Use official portals (UIDAI / Sarathi / NSP) directly."
            ),
            "docs": [],
            "inspector": f"[BACKEND GUARDRAIL] PII detected and masked before LLM/RAG pipeline.\n[JURISDICTION] {req.jurisdiction}",
        }

    # 2. RAG Retrieval
    topic_key, chunk = retrieve_chunk(req.query)

    # 3. Response Generation (V1 / V2 / V3)
    answer_text, sys_prompt = generate_llm_or_grounded_reply(req.query, req.variant, req.jurisdiction, chunk)

    inspector_log = (
        f"[BACKEND API] POST /api/chat (200 OK)\n"
        f"[VARIANT] {req.variant} | [JURISDICTION] {req.jurisdiction}\n"
        f"[RETRIEVED CHUNK] {topic_key}\n"
        f"[SYSTEM PROMPT] {sys_prompt}"
    )

    return {
        "type": "conflict" if topic_key == "expired_dl" and req.variant == "V3" else "bot",
        "title": f"{chunk['topic']} ({req.variant})" if chunk else f"JanSahayak Backend ({req.variant})",
        "body": answer_text,
        "docs": chunk["docs"] if chunk else [],
        "inspector": inspector_log,
    }
