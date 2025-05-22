# SNAFU: the Semantic Network and Fluency Utility

# What is SNAFU?

The semantic fluency task is frequently used in psychology by both reseachers
and clinicians. SNAFU is tool that helps you analyze fluency data. It can help
with:

<ul>
    <li>Counting cluster switches and cluster sizes</li>
    <li>Counting perseverations</li>
    <li>Detecting intrusions</li>
    <li>Calculating average age-of-acquisition and word frequency</li>
    <li>...more!</li>
</ul>

SNAFU also implements multiple network estimation methods which allow you to
perform network analysis on your data (see <a
href="https://link.springer.com/article/10.1007/s42113-018-0003-7">Zemla &
Austerweil, 2018</a>). These methods are implemented:

<ul>
    <li>U-INVITE networks</li>
    <li>Pathfinder networks</li>
    <li>Correlation-based networks</li>
    <li>Naive random walk network</li>
    <li>Conceptual networks</li>
    <li>First Edge networks</li>
</ul>

# How do I use SNAFU?

<p>SNAFU can be used as a Python library or as a stand-alone GUI. ThePython library is available here:</p>

<p><a href="https://github.com/AusterweilLab/snafu-py">https://github.com/AusterweilLab/snafu-py</a></p>

<p>Or install directly using git (auxilliary files are not included):</p>

<p><code class="highlighter-rouge">pip install git+https://github.com/AusterweilLab/snafu-py</code></p>

<p>The Github repository contains several demo files, and a tutorial covering some basic usage is available in <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7406526/">Zemla, Cao, Mueller, & Austerweil, 2020</a>

If you plan to use the `correlationBasedNetwork()` function, you will need to install the `planarity` package separately using `pip install planarity`

<p>A graphical front-end is also available, though it does not contain as many
features as the Python library. You can download it for macOS or Windows. Find
it here:</p>

<table>
    <th colspan=3>Mac</th>
    <tr><td><a href="https://alab.psych.wisc.edu/snafu/snafu-2.4.1-mac-x64.dmg">SNAFU 2.4.1 for macOS (latest version)</a></td></tr>
    <th colspan=3>Windows</th>
    <tr><td><a href="https://alab.psych.wisc.edu/snafu/snafu-2.2.0-win-x64.zip">SNAFU 2.2.0 for Windows</a></td></tr>
</table>

# How can I reference SNAFU?

The primary citation for SNAFU is:

> Zemla, J. C., Cao, K., Mueller, K. D., & Austerweil, J. L. (2020). SNAFU: The semantic network and fluency utility. Behavior Research Methods, 52, 1681-1699.

If using the English-language animal scheme (animals_snafu_scheme.csv), also cite:

> Troyer, A. K. (2000). Normative data for clustering and switching on verbal fluency tasks. Journal of Clinical and Experimental Neuropsychology, 22, 370-378.

> Hills, T. T., Jones, M. N., & Todd, P. M. (2012). Optimal foraging in semantic memory. Psychological Review, 119, 431-440.

The English-language foods scheme (foods_snafu_scheme.csv) should also cite the primary SNAFU publication and Troyer et al. (2000)

If using the English-language age-of-aquisition norms (kuperman.csv):

> Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. Behavior Research Methods, 44, 978-990.

If using the English-language word frequency norms (subtlex-us.csv):

> Brysbaert, M., & New, B. (2009). Moving beyond Kucˇera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English. Behavior Research Methods, 41, 977– 990.

If using the Dutch-language animal scheme (Dutch_animals_snafu_scheme.csv):

> Rofes, A., Beran, M., Jonkers, R., Geerlings, M.I., Vonk, J.M.J. (2023). What drives task performance in animal fluency in individuals without dementia? The SMART-MR study. Journal of Speech, Language, and Hearing Research (ASHA). Retrieved from: https://github.com/jmjvonk/2022_Rofes_SMART-MR/blob/main/Dutch_animals_snafu_scheme.csv

If using the Greek-language animal scheme (animals_Scheme_Greek_Karousou_v.01.2.csv):

> Karousou, A., Economacou, D., & Makris, N. (2023). Clustering and Switching in Semantic Verbal Fluency: Their Development and Relationship with Word Productivity in Typically Developing Greek-Speaking Children and Adolescents. Journal of Intelligence, 11(11), 209. https://doi.org/10.3390/jintelligence11110209

If using the Spanish-language animal scheme (animals_ESnoaccent_scheme.csv):

> Neergaard, K. D., Zemla, J. C., Lubrini, G., Periañez, J. A., Bernabéu, E., Ríos-Lago, M., ... & Ayesa-Arriola, R. (2025). Novel computational measure of semantic fluency performance associated with first-episode of psychosis. Psychiatry Research, 348, 116462.

The Mexican Spanish-language animal scheme (animals_snafu_mexican_spanish.csv) was adapted from the Spanish-language scheme above and provided by Yamilka Garcia Avila and Yaira Chamorro (Universidad of Guadalajara)

The Dutch-language schemes for bike parts, fruits, foods, transportation, and farm animals were provided by Adria Rofes (University of Groningen).

The Italian-language animal scheme (snafu_Italian_scheme_Costantini.csv) was provided by Sabia Costantini (Universität Potsdam)

If would like to contribute additional files to this repository, and/or would like to change the list citations associated with your work, please contact us.

# Need help?

Check out our <a href="https://groups.google.com/forum/#!forum/snafu-fluency">Google Group</a> that will be used for troubleshooting. If you have question or comment, e-mail the list at snafu-fluency [at] googlegroups [dot] com

