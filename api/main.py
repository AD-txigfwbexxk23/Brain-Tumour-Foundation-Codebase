from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

from services.predictor_service import get_predictor_singleton, PredictorInput

app = FastAPI(title="Brain Tumor Prediction API", version="1.0.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    default_values = {
        "Age": 50,
        "Tumor_Size": 5.0,
        "Tumor_Growth_Rate": "Moderate",
        "Symptom_Severity": "Moderate",
        "Tumor_Location": "Frontal",
        "MRI_Findings": "Abnormal",
        "Radiation_Exposure": "Medium",
    }
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "values": default_values,
            "errors": [],
            "result": None,
        },
    )


@app.get("/authors", response_class=HTMLResponse)
async def authors(request: Request):
    authors_list = [
        {"name": "Reyansh Badhwar", "role": "Author", "slug": "reyansh"},
        {"name": "Zehaan Walji", "role": "Author", "slug": "zehaan"},
        {"name": "Meherab Ali", "role": "Author", "slug": "meherab"},
        {"name": "Omar Negmeldin", "role": "Author", "slug": "omar"},
    ]
    return templates.TemplateResponse(
        "authors.html",
        {"request": request, "authors": authors_list},
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/model", response_class=HTMLResponse)
async def model_info(request: Request):
    return templates.TemplateResponse("model.html", {"request": request})


@app.get("/faq", response_class=HTMLResponse)
async def faq(request: Request):
    return templates.TemplateResponse("faq.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/changelog", response_class=HTMLResponse)
async def changelog(request: Request):
    return templates.TemplateResponse("changelog.html", {"request": request})


@app.get("/policy", response_class=HTMLResponse)
async def policy(request: Request):
    return templates.TemplateResponse("policy.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    Age: float = Form(...),
    Tumor_Size: float = Form(...),
    Tumor_Growth_Rate: str = Form(...),
    Symptom_Severity: str = Form(...),
    Tumor_Location: str = Form(...),
    MRI_Findings: str = Form(...),
    Radiation_Exposure: str = Form(...),
):
    predictor = get_predictor_singleton()
    input_payload = PredictorInput(
        Age=Age,
        Tumor_Size=Tumor_Size,
        Tumor_Growth_Rate=Tumor_Growth_Rate,
        Symptom_Severity=Symptom_Severity,
        Tumor_Location=Tumor_Location,
        MRI_Findings=MRI_Findings,
        Radiation_Exposure=Radiation_Exposure,
    )
    errors = predictor.validate_input(input_payload.dict())
    result = None
    if not errors:
        result = predictor.predict(input_payload.dict())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "values": input_payload.dict(),
            "errors": errors,
            "result": result,
        },
    )


@app.post("/api/predict", response_class=JSONResponse)
async def predict_api(payload: PredictorInput):
    predictor = get_predictor_singleton()
    errors = predictor.validate_input(payload.dict())
    if errors:
        return JSONResponse({"errors": errors}, status_code=400)
    return predictor.predict(payload.dict())


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


