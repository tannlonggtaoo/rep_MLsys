This is a repository for MLsys replication experiments.

## Quantization

### Post-training quantization (PTQ)

* Basic PTSQ implementation with PyTorch APIs

Resnet18 - CIFAR10

| bitW \ bitA | 2      | 3      | 4      | 5      | 6      |
| ----------- | ------ | ------ | ------ | ------ | ------ |
| 2bit        | 10.00% | 10.93% | 9.40%  | 9.86%  | 10.06% |
| 4bit        | 10.00% | 30.38% | 78.86% | 88.63% | 89.20% |
| 6bit        | 10.00% | 35.38% | 82.59% | 91.28% | 92.00% |

MobileNetV2 - CIFAR10

| bitW \ bitA | 2      | 3      | 4      | 5      | 6      |
| --------- | ------ | ------ | ------ | ------ | ------ |
| 2bit      | 10.00% | 10.00% | 10.00%  | 10.00%  | 10.00% |
| 4bit      | 10.23% | 11.08% | 27.42% | 77.10% | 88.22% |
| 6bit      | 10.18% | 11.18% | 36.22% | 82.56% | 90.54% |

### quantization-aware training (QAT)

* DOREFA-NET

  * SVHN  - SVHN (without extra data)
  * Resnet18 - CIFAR10
  * MobileNetV2 - CIFAR10

| bitW=bitA=bitG \ backbone | SVHN   | Resnet18 | MobileNetV2 |
| ------------------------- | ------ | -------- | ----------- |
| 2bit                      | 17.47% | ?        | ?           |
| 6bit                      | 86.30% | 89.60%   | 70.39%      |

*'?' = no reasonable converged results yet...*

For SVHN only :

| bitW - bitA - bitG | SVHN (7-layer CNN) |
| ------------------ | ------------------ |
| 2-2-6              | 83.9%              |
| 2-2-32             | 84.5%              |
| 6-6-2              | 19.6%              |
| 6-6-32             | 87.0%              |

* PACT 

  * Resnet18 - CIFAR10
  * MobileNet - CIFAR10

| bitW=bitA \ backbone | Resnet18 | MobileNetV2 |
| -------------------- | -------- | ----------- |
| 2bit                 | 93.84%   | 63.87%      |
| 3bit                 | 94.56%   | 85.15%      |
| 4bit                 | 94.16%   | 89.89%      |
| 5bit                 | 94.54%   | 88.61%      |
| 6bit                 | 94.53%   | 90.74%      |

* LSQ
  * Resnet18 - CIFAR10
  * MobileNet - CIFAR10

| bitW=bitA \ backbone | Resnet18 | MobileNetV2 |
| -------------------- | -------- | ----------- |
| 2bit                 | 91.4%    | 74.8%       |
| 5bit                 | 93.2%    | 91.5%       |
| 6bit                 | 93.9%    | 92.3%       |



## Pruning

*To be continued...*