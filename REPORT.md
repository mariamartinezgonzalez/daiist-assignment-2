# Assignment 2 Report
- **Name**: María Martínez González
- **Student ID**: 18752
- **Email**: mmg.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong
The model looks at a photo from a construction-site camera and decides whether each worker is wearing a hard hat. The site supervisor acts on it: if the model finds a bare head, the supervisor is alerted and goes to correct the situation before the worker keeps working.

A false negative is a worker without a helmet that the model misses. This is the worse mistake, because nobody is alerted and the worker is exposed to a head injury, a regulatory fine and possible stoppage of the work. A false positive is a worker with a helmet that the model flags by mistake. The cost is small: the supervisor walks over, sees the helmet and walks back, which costs a few minutes.

So I treat head as the positive class and I will favour recall: I prefer many false alarms over one missed violation. This is why the threshold will be low, and why uncertain cases will go to a person for review and not be decided automatically. The euro values for each error are in the costing section.

## Dataset
- Workspace: iote-7emb7
- Project: hard-hat-7785r
- Version: 1
- Link: https://universe.roboflow.com/iote-7emb7/hard-hat-7785r/dataset/1
- License: CC BY 4.0

Each image is a photo of a construction site or industrial area with one or more workers in it. The dataset has two classes. helmet marks a worker who is wearing a hard hat. head marks a bare head, so a worker with no hard hat. In my problem head is the positive class, because it is the safety violation that a supervisor needs to be told about.

I chose this dataset because the images look like what a site camera would really see: workers at different distances, angles and lighting, indoors and outdoors. The head class is exactly the violation I want to catch, so the dataset matches the decision and does not just look similar.

The version I use is v1 with 7,035 images, resized to 640x640 with no augmentations. The split is 4,221 train (60%), 2,111 validation (30%) and 703 test (10%). I did not fork the dataset, so anyone can download the same version.

## What the dataset's composition implies
Counting the boxes in the label files, the train split has 4,131 head and 11,717 helmet instances, the validation split has 1,838 and 6,044, and the test split has 708 and 1,986. So head is about a quarter of all boxes. That is plenty of examples for both classes, so I think the per-class numbers can be trusted. The test split is still small though (703 images, 146 of them with a bare head), so my results could move a few points with different images.

The objects are small. The median head box covers only 0.6% of the image and the median helmet box 0.8%. This is why I train at 640 pixels and not smaller: at low resolution these boxes would be just a few pixels wide and the model would miss many of them.

The test split has 21% of images with a bare head, which is much higher than what I would expect on a real site (about 5%). That means precision and cost numbers measured on the test set will look worse than in real life, so I correct for it with the base rate in the costing section. Also, some images look like press photos with watermarks, not like fixed cameras, so I can't conclude the model works on a specific camera at a specific site without testing on that site's footage.

## Finetuning and baselines
I use object detection, not classification, because one image can contain several workers, some with a helmet and some without, and the supervisor needs to know that at least one bare head is there. I finetune yolo11n (the smallest YOLO11 model) starting from its pretrained COCO weights. I chose the small one because it is fast to train and run, and because I wanted to see first whether a small model is already good enough before paying for a bigger one.

For the comparison I use three things on the same test split: the finetuned model, the pretrained COCO model used zero-shot, and a trivial baseline that predicts nothing. COCO has no head or helmet class, so I expect it to detect none of them and score zero.

TODO: results table (mAP50, mAP50-95 and cost per 1,000 cases for each of the three), where I trained, the settings I used, and whether the gap is as big as I expected.

## Per-class evaluation and failure modes
TODO: precision, recall and AP50 for head and for helmet, and the confusion matrix. Say which class carries the headline number, which one is weaker, and what the model confuses with what. Positive class: head.

## Cost matrix and operating point
All costs are per case, where one case is one camera image. They are my own assumptions, not measured values.

A false positive costs 2.50 euros. The supervisor stops what they are doing, walks over and sees the worker has a helmet. I assumed 5 minutes at a labour cost of 30 euros per hour. A true positive costs the same 2.50 euros, because the walk is identical and the supervisor then fixes the problem. A true negative costs 0.

A false negative costs 300 euros. This is the expected cost of leaving a bare head unaddressed, not the cost of an injury itself. I assumed a serious incident (medical costs, work stoppage, possible fine) costs about 30,000 euros, and that a missed bare head in one image leads to such an incident about 1% of the time. 30,000 x 1% = 300. This is the number I trust least. If the incident cost or the probability falls by a factor of ten, a miss is only worth about 30 euros and the best threshold moves up.

I assume a base rate of 5%: on a real site only about 1 image in 20 shows a worker without a helmet. The test split has 21%, so it is enriched and I use 5% for the costing instead.A person reviewing one image costs 0.25 euros, about 30 seconds at 30 euros per hour.Because a miss costs about 120 times more than a false alarm, a case should be flagged as soon as the probability of a violation is above roughly 1%. So I expect a low threshold.

TODO: the cost against threshold curve, the threshold I chose and its cost per 1,000 cases.

## What a person still has to check
TODO: my two thresholds (pass / human review / fail), the share of cases sent to a person, cases per day, minutes per check, how many people that needs and whether it is realistic. Then compare four options: do nothing, a person checks everything, the model decides alone, and model plus human review.

## Limitations & next steps
- **The test split is not like a real site.** It has 21% bare heads against my assumed 5%, and some images are press photos with watermarks. With more time I would test on footage from a real site camera and measure the real base rate.
- **The objects are tiny.** The median box is under 1% of the image, so I expect missed heads far from the camera. I would try a bigger model (yolo11s) or a higher image size, and compare.
- **One case is one image.** I flag an image if it has at least one bare head, so I don't track whether a worker is still without a helmet in the next frame. With more time I would combine several frames before alerting.
- **The costs are assumptions.** The 300 euros for a missed head depends on a guessed 1% chance of an incident. I would replace it with real incident and insurance data from the company.

## Generative AI use disclosure
I used Claude (Anthropic) as an assistant. It helped me set up the repository and Codespace, fix installation errors.