# home-credit-risk-scoring
End-to-end credit risk scoring project using the Home Credit dataset, including preprocessing, modeling, calibration, explainability and business-oriented evaluation.

## Business problem 

This project develops an intepretable machine learning pipeline to estimate a loan applicant experiences payment difficulties.

The prediction is made at application time using only information assumed to be available before the lending decision.

The model is intended as a decision-support and risk-ranking tool, not as a fully automated approval system.

The target provided Home Credit is a proxy for payment difficulty and should not be intepreted as a regulatory 12-month probability of default.

## Population studied 

The studied population consists of loan applicants contained in the Home Credit dataset.

Each observation corresponds to one credit application submitted by a customer.

## Unit of observation 

One observation corresponds to one loan application.

## Target event 

The target event is the payment difficulty indicator provided by Home Credit.

A target value of 1 indicates payment difficulties according to the dataset definition.

A target value of 0 indicates no payment difficulties.

# Prediction time 

Predictions are made at loan application time, before any lending decision is taken.

Only information available at application time shoud be used.

## Intended use

The model is designed to support credit analysts by ranking applicants according to estimated payment risk.

It may also be used to support portfolio segmentation and threshold analysis.

The final lending decision remains under human supervision.

## False positives and false negatives

False positives correspond to applicants classified as risky although they would not experience payment difficulty. Their consequence is the rejection of potentially profitable customers.

False negatives correspond to risky applicants classified as safe. There consequence is an increased credit risk and potential financial losses.

## Out-of-scope uses

This model is not intented to :

- replace human credit analysts;
- estimate a regulatory probability of default;
- make legally binding lending decisions;
- evaluate customers after loan origination;
- assess fairness or regulatory compliance.