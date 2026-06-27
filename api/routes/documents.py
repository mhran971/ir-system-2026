from fastapi import APIRouter, HTTPException
from api import services
from api.models import DocumentResponse

router = APIRouter(tags=["documents"])


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str):
    """Retrieve the full text of a document by its ID."""
    svc = services.get_simple()
    if svc is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    doc = svc.document_store.get_doc(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    return DocumentResponse(doc_id=doc["doc_id"], text=doc.get("text", ""))
