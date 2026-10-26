from fastapi import APIRouter

router = APIRouter(prefix="/card-flag", tags={"Health"})

@router.get("/health")
def health_check():
    return {"status": "API is healthy and running."}
