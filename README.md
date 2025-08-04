# Classificador de Sentimentos com BERT + FastAPI

Projeto desenvolvido para desafio prático de **Arquiteto de Machine Learning Sênior**, com foco no ciclo completo de um sistema de Machine Learning — desde o fine-tuning até a disponibilização via API RESTful.

---

## 📁 Estrutura do Projeto

```
ml-sentiment/
├── training/
│   └── fine_tune.py          # Script de fine-tuning com BERT
├── api/
│   └── main.py               # API RESTful com FastAPI
├── models/
│   └── sentiment_model/      # Artefatos salvos (modelo + tokenizer)
├── requirements.txt
├── README.md
```

---

## 📌 Dataset

Utilizei o dataset público [`b2w-reviews01`](https://github.com/americanas-tech/b2w-reviews01), contendo mais de 15 mil avaliações de produtos. O campo `overall_rating` foi mapeado em:

- 1 ou 2 → **Negativo** (`0`)
- 4 ou 5 → **Positivo** (`1`)
- 3 → Removido por ser neutro

---

## 🧪 Fine-tuning com BERT

Modelo base: [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

### 📥 Executar fine-tuning

```bash
python training/fine_tune.py
```

O modelo treinado será salvo em:

```
./models/sentiment_model/
```

---

## 🚀 API RESTful com FastAPI

Após o treinamento, disponibilizamos o modelo via uma API.

### 📁 Arquivo: `api/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

MODEL_PATH = "./models/sentiment_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio.")

    inputs = tokenizer(
        input_data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs).item()
        score = round(probs[0][pred_class].item(), 4)

    label = "positivo" if pred_class == 1 else "negativo"
    return {"sentiment": label, "score": score}
```

---

### ▶️ Rodar a API localmente

```bash
uvicorn api.main:app --reload
```

Acesse em: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 🧪 Exemplo de chamada com `curl`

```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"text": "Produto excelente, chegou antes do prazo!"}'
```

**Resposta:**
```json
{
  "sentiment": "positivo",
  "score": 0.9821
}
```

---

## 💡 Decisões Técnicas

- **Modelo**: `bert-base-portuguese-cased` foi escolhido por ser um modelo BERT treinado com corpus em português, garantindo melhor compreensão dos textos da base.
- **Pré-processamento**: Filtrei textos vazios/nulos e removi avaliações neutras (nota 3), para melhor desempenho binário.
- **Treinamento**: Realizado com `Trainer` da Hugging Face e métricas de `accuracy` e `f1`.
- **API**: Implementada com FastAPI por sua performance e documentação automática.

---

## 📈 Possível Evolução para Produção

- Servir com **TorchServe** ou **Triton** para maior escalabilidade.
- Containerização com **Docker**.
- Monitoramento com Prometheus + Grafana.
- Logging estruturado.
- Versionamento de modelo com MLflow.
- Load balancing com Nginx ou Kubernetes.

---

## 📦 Requisitos

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```
