# Valuing Player Actions in Ice Hockey: Comparing VAEP and Expected Threat

All the files used for the Thesis by Cas Gerritsen. This thesis addressed the following research question: "How effectively can ice hockey player actions be valued using a VAEP-inspired model compared to an Expected Threat model?" The study explored how player actions can be assigned value using the VAEP and xT frameworks and examined how these models perform relative to one another.

The folder **main_pipeline** contains the key scripts, organized by the following steps:

-*sync_events.py* includes the code used to process the input files. The input is an IDF file with tracking data, and the output is a dataframe containing all actions with their associated features as columns.

-*actions_to_gamestates.ipynb* implements the left-to-right transformation, labels scoring and conceding actions, and creates the game states.

-*evaluation.cat.ipynb* handles feature scaling, model training, testing, validation, and evaluation.

-*xT.ipynb* contains the entire Expected Threat pipeline. It uses the same data generated in *actions_to_gamestates.ipynb*, but works with a modified dataframe that does not include actions as game states.

Each of the main files begins with a short explanation at the top (in addition to inline comments) describing the contents and purpose of the script.

For questions about code or data, you can always send an email to cas.gerritsen@student.uva.nl or cas16015@gmail.com. Feel free to reach out!
