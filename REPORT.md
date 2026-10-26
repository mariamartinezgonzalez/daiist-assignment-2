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

Each image is a photo of a construction site or industrial area with one or more workers in it. The dataset has two classes. Helmet marks a worker who is wearing a hard hat. Head marks a bare head, so a worker with no hard hat. In my problem head is the positive class, because it is the safety violation that a supervisor needs to be told about.

I chose this dataset because the images look like what a site camera would really see: workers at different distances, angles and lighting, indoors and outdoors. The head class is exactly the violation I want to catch, so the dataset matches the decision and does not just look similar.

The version I use is v1 with 7,035 images, resized to 640x640 with no augmentations. The split is 4,221 train (60%), 2,111 validation (30%) and 703 test (10%). I did not fork the dataset, so anyone can download the same version.

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
