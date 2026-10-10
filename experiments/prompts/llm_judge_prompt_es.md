# LLM-as-a-Judge rubric

Evaluate each response using only the supplied question, ground truth, authoritative evidence, and generated response

Do not use external knowledge

Do not penalize a short response if it completely answers the question

For open-ended questions, require all essential requested elements

## factual_correctness

- 0 incorrect or contradicts the evidence
- 1 partially correct or contains a minor error
- 2 fully correct

## completeness

- 0 misses the central answer
- 1 answers the main point but omits important information
- 2 covers all essential elements

## faithfulness_to_evidence

- 0 contradicts or is substantially outside the evidence
- 1 mostly grounded but includes a minor unsupported detail or imprecision
- 2 fully supported by the evidence

## unsupported_information

- 0 none
- 1 minor unsupported addition
- 2 substantial unsupported addition

## overall_verdict

- PASS when factual_correctness=2, completeness=2, faithfulness_to_evidence=2, and unsupported_information=0
- PARTIAL when the answer is essentially correct but has a minor error, omission, or limited unsupported addition
- FAIL when the central answer is wrong, contradicted, critically incomplete, or substantially unsupported

Return one evaluation record for each anonymized response A, B, and C
