from fastapi import FastAPI
import json
from pydantic import BaseModel
from diseases.wrapup import *
import numpy as np
app = FastAPI()

class Symptoms(BaseModel):
    symptoms:list[int]

@app.get("/")
def root():
    covid = ["Does the patient have breathing problem ? ","Does the patient have fever ?","Does the patient have dry cough ?","Does the patient have any record of asthma ?","Does the patient have any records of chronic lung disease ?"]
    return covid

@app.post("/")
def isInfected(data : Symptoms):
    test = data.symptoms

    if result([test])[0] == 1 :
        return 1
    return 0