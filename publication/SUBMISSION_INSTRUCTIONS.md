# IEEE RA-L Submission Instructions

## Step 3: Upload to IEEE ScholarOne

### Access
- **CORRECT URL**: https://ras.papercept.net/journals/ral
- Login with your RAS PIN number and password
- If no PIN, register at: https://ras.papercept.net/journals/ral/scripts/login.pl

### Manuscript Submission

**Paper Type**: Regular Paper (not Review or Correspondence)

**Title**: 
Hybrid Inverse Kinematics Ensemble with Learned Uncertainty Estimation for Robotic Manipulation

**Authors**:
1. Mulham Fetna (Primary)
   - Department of Computer Science and Engineering
   - University of Bologna, Italy
   - Email: mulham.fetna@studio.unibo.it
   
2. Luca Ricci (Co-author)
   - Department of Computer Science
   - University of Tuscia, Italy

**Abstract** (copy from paper):
We present a hybrid inverse kinematics (IK) ensemble that combines multiple solver strategies with learned uncertainty estimation. The approach integrates analytical (Damped Least Squares), learned (Neural Network), and meta-learning (ensemble weighting) components to achieve robust performance across diverse workspace regions. Unlike single-solver approaches, our method quantifies prediction uncertainty and adaptively weights solver contributions based on confidence. Experiments on a 6-DOF UR5-like manipulator demonstrate 100% success rate on random targets with 5ms average solve time, outperforming traditional analytical methods. The ensemble achieves 86.7% overall success rate across challenging scenarios including singularities, joint limits, and workspace boundaries.

**Keywords**: 
Inverse Kinematics, Neural Networks, Ensemble Learning, Robotic Manipulation, Uncertainty Quantification

**Technical Area**: Robotics and Automation

**Suggested Reviewers** (optional):
- None (allow editor assignment)

### File Upload

1. **Main Manuscript**: Upload `paper.pdf` from `publication/paper.pdf`
2. **Figures**: Include all figures in the manuscript
   - system_diagram.png (Figure 1)
   - benchmark_plot.png (Figure 2)  
   - comparison_plot.png (Figure 3)

### Additional Required Information

**Cover Letter**: (Optional) - highlight novelty: hybrid ensemble + learned weighting

**Prior Work**: Cite related work, no prior submission to RA-L

**Animal/Human Subjects**: Not applicable

---

## Step 4: Conflict of Interest Form

### Generate Form Content

**COI Statement for IEEE RA-L Submission**:

```
The authors declare no conflict of interest.

All authors confirm that this work is original and has not been submitted 
to any other journal or conference simultaneously.

This work was supported by the University of Bologna and University of 
Tuscia research programs.
```

### How to Submit:
1. During submission, select "No" for "Do you have any conflicts to declare?"
2. Or download IEEE COI form from: https://www.ieee.org/publications/rights/conflict-of-interest.html

---

## Step 5: Code Availability Statement

### Statement:

```
The source code and data for this work is publicly available at:
https://github.com/molhamfetnah/ik-uncertainty-learning

The repository includes:
- Hybrid IK Ensemble implementation (Python/NumPy)
- Damped Least Squares solver
- Neural Network solver (MLP, trained on 4640 samples)
- Benchmark suite (7 scenarios, 60 targets)
- Stress tests (7 tests)
- Training data generation scripts
- All experimental results

The code runs without external deep learning frameworks (PyTorch/TensorFlow),
using only NumPy for neural network implementation.

License: MIT License
```

### How to Include:
- Add this statement to the end of the paper before references
- Or provide in "Supplementary Materials" during submission
- Link in submission comments

---

## Submission Checklist

- [x] Paper PDF (5 pages)
- [x] Figures (3 images)
- [x] Abstract (under 250 words)
- [x] Keywords (4-6)
- [x] Author affiliations
- [x] COI statement prepared
- [x] Code availability statement prepared
- [x] GitHub repository ready: https://github.com/molhamfetnah/ik-uncertainty-learning

## Next Steps

1. Go to **https://ras.papercept.net/journals/ral**
2. Login with RAS PIN (or register new PIN)
3. Start new submission
4. Fill in all information above
5. Upload paper.pdf
6. Submit

Expected confirmation: 2-3 business days
Review timeline: 3-6 months