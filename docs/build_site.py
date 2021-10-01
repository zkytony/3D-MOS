# Converts the README.md to HTML
import markdown
import os

README_PATH = "../README.md"
OUTPUT_DIR = "html"

INSERT =\
"""
* **ArXiv**: [https://arxiv.org/abs/2005.02878](https://arxiv.org/abs/2005.02878)
* **PDF**: [https://arxiv.org/pdf/2005.02878.pdf](https://arxiv.org/pdf/2005.02878.pdf)
* **Github**: [https://github.com/zkytony/3D-MOS](https://github.com/zkytony/3D-MOS)
* **Robot demo**: [https://www.youtube.com/watch?v=oo-wrL0ta6k](https://www.youtube.com/watch?v=oo-wrL0ta6k)
* **Website**: [https://zkytony.github.io/3D-MOS/](https://zkytony.github.io/3D-MOS/)
* **Blog** [https://h2r.cs.brown.edu/object-search-in-3d/](https://h2r.cs.brown.edu/object-search-in-3d/)



## Demo
<div class="row ml-2 mt-3 mb-5">
<iframe width="560" height="315" src="https://www.youtube.com/embed/oo-wrL0ta6k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>


## Talk
<div class="row ml-2 mt-3 mb-5">
<iframe width="560" height="315" src="https://www.youtube.com/embed/5G09TRepJLY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Overview
Robots operating in human spaces must find objects such as glasses, books,
or cleaning supplies that could be on the floor, shelves, or tables. This search
space is naturally 3D.

When multiple objects must be searched for, such as a cup and a mobile phone, an
intuitive strategy is to first hypothesize likely search regions for each target
object based on semantic knowledge or past experience, then search carefully
within those regions by moving the robot’s camera around the 3D environment. To
be successful, it is essential for the robot to produce an efficient search
policy within a designated search region under limited field of view (FOV),
where target objects could be partially or completely blocked by other
objects. In this work, we consider the problem setting where a robot must search
for multiple objects in a search region by actively moving its camera, with as
few steps as possible.

Searching for objects in a large search region requires acting over long
horizons under various sources of uncertainty in a partially observable
environment. For this reason, previous works have used Partially Observable
Markov Decision Process (POMDP) as a principled decision-theoretic framework for
object search. However, to ensure the POMDP is manageable to solve, previous
works reduce the search space or robot mobility to 2D, although objects exist in
rich 3D environments. The key challenges lie in the intractability of
maintaining exact belief due to large state space, and the high branching factor
for planning due to large observation space.

In this paper, we present a POMDP formulation for multi-object search in a 3D
region with a frustum-shaped field-of-view. To efficiently solve this POMDP, we
propose a multi-resolution planning algorithm based on online Monte-Carlo tree
search. In this approach, we design a novel octree-based belief representation
to capture uncertainty of the target objects at different resolution levels,
then derive abstract POMDPs at lower resolutions with dramatically smaller state
and observation spaces.

Evaluation in a simulated 3D domain shows that our approach finds objects more
efficiently and successfully compared to a set of baselines without resolution
hierarchy in larger instances under the same computational requirement.

Finally, we demonstrate our approach on a torso-actuated mobile robot in a lab
environment. The robot finds 3 out of 6 objects placed at different heights in
two 10m2 x 2m2 regions in around 15 minutes.
"""

# convert README as html
with open(README_PATH) as f:
    md = f.read()
    md = md.replace("<!-- #<># -->", INSERT)

    content = markdown.markdown(md, extensions=["fenced_code"])

    # Replace asset paths
    content = content.replace("docs/html/figs", "figs")
    # Replace <p><code> with <pre><code>
    content = content.replace("<p><code>", "<pre><code>")
    content = content.replace("</code></p>", "</code></pre>")

# Load template
with open("_template.html") as f:
    template = f.read()
    html = template.replace("#{}#", content)

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(html)
