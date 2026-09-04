# Merchant Digital Twin — Award Edition

Track 01 prototype: **simulate before you act**.

## Included features
- Counterfactual what-if simulator
- Customer segment digital twin
- Profit/revenue/orders/conversion optimizer
- Monte Carlo uncertainty and probability of positive profit
- Natural-language scenario parser
- AI experiment designer
- Root-cause/risk radar
- Explainable “Why?” evidence
- Digital Twin confidence score + model validation metrics
- Guardrails + merchant approval workflow
- Simulation receipt/audit ID + SQLite history
- What-if strategy comparison
- Bounded autonomous experiment agent (recommendation only; no production action)

## Local VS Code
Terminal 1:
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Terminal 2:
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## One-command style with Docker
```powershell
docker build -t merchant-digital-twin .
docker run --rm -p 8000:8000 merchant-digital-twin
```
Open http://localhost:8000

Synthetic data only. Financial values are illustrative. No production action is executed without approval.
