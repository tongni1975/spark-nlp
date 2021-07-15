---
layout: model
title: Pipeline for Adverse Drug Events
author: John Snow Labs
name: explain_clinical_doc_ade
date: 2021-07-15
tags: [licensed, clinical, en, pipeline]
task: [Named Entity Recognition, Text Classification, Relation Extraction, Pipeline Healthcare]
language: en
edition: Spark NLP for Healthcare 3.1.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pipeline for Adverse Drug Events (ADE) with `ner_ade_biobert`, `assertion_dl_biobert`, `classifierdl_ade_conversational_biobert`, and `re_ade_biobert` . It will classify the document, extract ADE and DRUG clinical entities, assign assertion status to ADE entities, and relate Drugs with their ADEs.

Pipeline components:
- DocumentAssembler
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
- NerConverter
- AssertionDLModel
- SentenceEmbeddings
- ClassifierDLModel
- RelationExtractionModel

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_ade_en_3.1.2_3.0_1626380200755.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

res = pipeline.fullAnnotate('I am happy that now am using simvistatin. My hip pain is now gone which was caused by my previous drug, lipitor.')


```
```scala
val era_pipeline = new PretrainedPipeline("explain_clinical_doc_era", "en", "clinical/models")

val result = era_pipeline.fullAnnotate("""I am happy that now am using simvistatin. My hip pain is now gone which was caused by my previous drug, lipitor.""")(0)

```
</div>

## Results

```bash
|    | chunk1                        | entitiy1   | chunk2      | entity2 | relation |
|----|-------------------------------|------------|-------------|---------|----------|
| 0  | hip pain is now gone          | ADE        | simvistatin | DRUG    |        0 |
| 0  | hip pain is now gone          | ADE        | lipitor     | DRUG    |        1 |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_ade|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings
- SentenceEmbeddings
- ClassifierDLModel
- MedicalNerModel
- NerConverterInternal
- PerceptronModel
- DependencyParserModel
- RelationExtractionModel
- NerConverterInternal
- AssertionDLModel