# Resume after environment loss

Completed training: R01 v1 374890167, R02 v1 374890664, S01 v2 374897225. S01 v1 failed before training (logged separately). All completed outputs are durable on Kaggle. Maximum concurrent saved runs remains 2.

No Kaggle reauthentication is needed unless Kaggle actually shows a sign-in wall. Account is quannguyen0501, already verified by the user. GitHub is connected as nmquan22.

Do not restart successful training. Download own R01 and S01 best.pt/tokenizer/report from Kaggle Output. Recreate frozen inference bundle with code, train/dev, stripped test inputs, and own checkpoints. Never include test VA/category/opinion gold in inference input. Task1 keeps Aspect list; tasks2/3 keep only ID/Text. Use organizer commit bdc93be1224106ae7d3eb95739c02a76ed4ae8a1.

Frozen recipe: Task1 equal average C02 and R01 rounded predictions. Task2/3 S01 epoch30, span threshold .7, link threshold .7. Reproduce dev predictions first, run test inference once, download predictions, score locally with organizer test gold. Do not change model/thresholds based on test scores.

Dataset currently verified at private version1 only. New version upload and E01 saved run were NOT verified before environment and browser transport loss. Do not report test scores or claim a running E01 job. Full S01 predictions/epochs were downloaded locally, but only report/scorer summary/diagnostics were recoverable through cloud GitHub after execution loss.
