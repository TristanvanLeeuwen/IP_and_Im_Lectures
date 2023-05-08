# Welcome to Inverse Problems and Imaging

```{note}
This is version 1 of the lecture notes.

<script type="text/javascript">
function lastModified() {
    var modiDate = new Date(document.lastModified);
    var showAs = modiDate.getDate() + "-" + (modiDate.getMonth() + 1) + "-" + modiDate.getFullYear();
    return showAs
}
document.writeln("<div style='font-size: 14px;'>");
document.write("Last updated on " + lastModified());
document.writeln("</div>");
</script>

```

In these lectures we introduce basic concepts related to inverse problems and visit various viewpoints on inverse problems. The main recurring themes are

* Modelling -- *how do we formulate a real-world problem in the language of mathematics*
* Analysis -- *is the corresponding mathematical problem well-posed*
* Solution strategies -- *how do we solve the problem numerically*
* Applications -- *where do inverse problems arise*

Each chapter contains exercises, assignments and example code in Python. The exercises allow you to practise with the basic concepts and techniques introduced in the chapter. Hints and answers to the exercises are provided in dropdown boxes. The assignments are more challenging and open-ended. They require you to integrate concepts and techniques learned in the course.

To get familiar with Python, we highly recommend [these lecture notes](https://scipy-lectures.org/).

## A tour through the book in 10 Lectures

* **Lecture 1 - [What is an inverse problem?](./what_is)** In this lecture we give an overview of the course and introduce the basic nomenclature.
* **Lecture 2 - [Discrete inverse problems](./discrete_ip_regularization).** In this lecture we treat linear inverse problems in $\mathbb{R}^n$. It introduces most of the basic concepts that will be treated in more generality later.
* **Lecture 3 - [Application: Image processing](./image_processing).** In this lecture we give an overview of inverse problems arising in various imaging applications like microscopy.
* **Lecture 4 - [Inverse problems in function spaces](ip_function_spaces).** Here, we treat the concepts of the previous lectures in a more general setting.
* **Lecture 5 - [A statistical perspective](./statistical_perspective).** We revisit the finite-dimensional setting of lecture 2 and introduce a statistical (Bayesian) perspective on inverse problems.
* **Lecture 6 - Application: [Computed Tomography](tomography).** In this lecture, we treat the tomographic inverse problem, which arises in various applications ranging from electron microscopy to medical imaging.
* **Lecture 7 - [Variational formulations](variational_formulations).** In this lecture we treat well-posedness and optimality conditions of variational problems arising in inverse problems.
* **Lecture 8 - [Numerical optimisation](numerical_optimisation).** In this lecture we discuss various methods for solving variational problems.
* **Lecture 9 - Application: [Wavefield imaging](wavefield_imaging).** In this lecture we discuss various inverse problems involving the wave equation.
* **Lecture 10 - Recent advances.** In this final lecture we discuss some recent advances, such as Compressed Sensing or the use of Machine Learning to solve inverse problems.

## Using the notes

Do you want to use these notes to teach a class on inverse problems or for self-study?
Please go ahead! These lecture notes are freely available and may be distributed and modified under a [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) license.

## Contributing to the notes  

Did you spot a mistake? Do you want clarify something or add an example or exercise? Click the Github logo at the top of the page to suggest edits! All files are written in [MyST Markdown](https://jupyterbook.org/reference/cheatsheet.html).

## Acknowledgements

Special thanks to Sven Dummer for proofreading and suggesting corrections.
