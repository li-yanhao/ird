# Test on a large-scale dataset

## Prepare

Download the Raise-1k dataset

Create resampled images at different factors and JPEG-compressed at different qualities.

## Run

```
# Evaluate the method with a specific configuration
python test.py --config config/xxx.yaml
```


## Result

Check the `plot.py` script for evaluation with different metrics:

- Accuracy at FPR=1%
- ROC curve



