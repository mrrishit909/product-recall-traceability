# Evaluation

Three plants the demo never uses (seeds 21, 31, 41), each with sixty days of production and the same supplier notice:
lot B-177, five days of date codes. Every strategy is built from the records alone; the simulation truth (which units really hold a
capacitor from the window's reels) scores them. Reproduce with `python -m rc.evaluate`.

## Recall strategies against the truth

| Plant | Strategy | Units recalled | Cost | Affected units recalled | Affected units missed | Unaffected units recalled |
|---|---|---|---|---|---|---|
| 21 | Truth | 3,243 | $422k | | | |
| 21 | Model: minimum defensible set | 3,296 | $428k | 3,242 | 1 | 54 |
| 21 | MES serial genealogy (deterministic) | 3,248 | $385k | 2,757 | 486 | 491 |
| 21 | Whole work orders that used a suspect reel | 3,952 | $468k | 2,766 | 477 | 1,186 |
| 21 | Every unit built while B-177 was on a line | 8,412 | $1,096k | 2,766 | 477 | 5,646 |
| 31 | Truth | 3,268 | $493k | | | |
| 31 | Model: minimum defensible set | 3,320 | $499k | 3,268 | 0 | 52 |
| 31 | MES serial genealogy (deterministic) | 3,273 | $496k | 3,228 | 40 | 45 |
| 31 | Whole work orders that used a suspect reel | 4,450 | $677k | 3,238 | 30 | 1,212 |
| 31 | Every unit built while B-177 was on a line | 5,707 | $823k | 3,235 | 33 | 2,472 |
| 41 | Truth | 2,816 | $479k | | | |
| 41 | Model: minimum defensible set | 2,883 | $488k | 2,816 | 0 | 67 |
| 41 | MES serial genealogy (deterministic) | 3,285 | $566k | 2,781 | 35 | 504 |
| 41 | Whole work orders that used a suspect reel | 4,235 | $727k | 2,782 | 34 | 1,453 |
| 41 | Every unit built while B-177 was on a line | 7,070 | $1,281k | 2,787 | 29 | 4,283 |

## Why the MES gets it wrong

| Plant | Missed: cause (units) | Recalled without cause: reason (units) |
|---|---|---|
| 21 | lot code misread or unread at receiving (457); rework ticket without the replacement board's serial (19); reel scanned late or not at all at a splice (10) | reel scanned late or not at all at a splice (455); rework ticket without the replacement board's serial (36) |
| 31 | rework ticket without the replacement board's serial (30); reel scanned late or not at all at a splice (10) | rework ticket without the replacement board's serial (44); reel scanned late or not at all at a splice (1) |
| 41 | rework ticket without the replacement board's serial (28); reel scanned late or not at all at a splice (7) | reel scanned late or not at all at a splice (466); rework ticket without the replacement board's serial (38) |

## Calibration of the unit probabilities (three plants pooled)

| Predicted band | Units | Mean predicted | Share truly affected |
|---|---|---|---|
| 0.0001-0.05 | 157 | 0.007 | 0.013 |
| 0.05-0.25 | 103 | 0.189 | 0.175 |
| 0.25-0.75 | 123 | 0.544 | 0.593 |
| 0.75-0.95 | 42 | 0.867 | 0.929 |
| 0.95-0.9999 | 6,069 | 0.999 | 1.000 |

## What the record models learned

| Plant | Lot-code character error (simulator 0.016) | Scan delay, line 3, minutes (simulator: 7 on line 3, 10 on lines 1-2) | Boards per reel (simulator about 490) | Splices observed |
|---|---|---|---|---|
| 21 | 0.0080 | 7.9 | 490.3 ± 3.2 | 28 |
| 31 | 0.0230 | 8.9 | 490.3 ± 3.4 | 28 |
| 41 | 0.0116 | 7.5 | 489.9 ± 2.9 | 28 |

## Policy: how much risk to leave out (three plants summed)

| Max expected escapes (per plant) | Exclude above | Units recalled | Cost | Affected missed | Expected escapes (model, sum) |
|---|---|---|---|---|---|
| 0.05 | 0.001 | 9,561 | $1,425k | 1 | 0.03 |
| 0.5 | 0.01 | 9,499 | $1,415k | 1 | 0.28 |
| 2.0 | 0.05 | 9,465 | $1,410k | 3 | 1.11 |
| 5.0 | 0.2 | 9,418 | $1,404k | 10 | 6.88 |

## When the model's assumptions are wrong (three plants summed)

The model assumes three things it cannot check: stores hand out the oldest spares, lines 1 and 2 scan like line 3, and the advance ship notices count reels correctly. Each row breaks one in the simulator and leaves the model alone.

| Simulator | Model units | Model affected missed | Expected escapes (model) | MES affected missed |
|---|---|---|---|---|
| As built (the rows above) | 9,499 | 1 | 0.28 | 561 |
| Reworks take any spare on the shelf, not one of the five oldest | 9,324 | 69 | 0.26 | 533 |
| Lines 1-2 scan delays three times line 3's | 9,479 | 19 | 0.27 | 592 |
| Every notice claims one reel more than arrived | 10,782 | 1 | 0.70 | 561 |

Run time 2 s on one core.
