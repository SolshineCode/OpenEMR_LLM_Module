# OpenEMR_LLM_Module

//Work in progress, not tested for professional deployment, yet//

PRs welcome 🤗 

Currently being updated to run via Ollama

An OpenEMR module creating a new tab with an LLM downloaded from Hugging Face and ran locally (to assure patient data privacy compliance) and able to analyze patient data conversationally.

## Medical Assistant LLM Module for OpenEMR

This project is a custom module for OpenEMR that integrates a Hugging Face language model to assist with medical tasks. The module adds a new tab to the OpenEMR interface, where users can interact with the language model.

It is currently in Beta Testing and being an independant open source contribution it is not affiliated with the OpenEMR organization itself. Please test out in your non-live systems and provide feedback in the comments or by DMing @ SolshineCode on Github or Solshine on Hugging Face.

## Features

- Integration with a Hugging Face language model: The module uses the `transformers` library to download and use a language model locally.
- New tab in OpenEMR: The module adds a new tab to the OpenEMR interface, where users can interact with the language model. The tab includes a prompt window where users can input a prompt, and an answer window where the model's response is displayed.
- Connection to OpenEMR patient data: The module can retrieve patient data from OpenEMR and pass it to the language model (patient data retrieval still needs some development.)
- The implementation of patient data retrieval and feedback functionality requires further development.

## Code Overview

The module consists of several parts:

- `module.info`: This file describes the module and is required by OpenEMR.
- `llm.py`: This Python script uses the `transformers` library to download and use a language model. It defines a function `generate_response` that takes a prompt as input, encodes it into a format that the model can understand, generates a response, and decodes the response into text.
- `menu_data.json`: This file defines a new menu item that links to the module's GUI.
- `module.config.php`: This file is a basic config for the custom module. This is eventually where you can specify the Hugging Face model to use as the assistant chatbot (pending this function, which is available to specify for now in the llm.py file)
- `llm.php`: This PHP file serves the module's GUI. It includes a form for the user to input a prompt, and a display area to show the model's response. It uses AJAX to send the prompt to a PHP script that interacts with the model and returns the response.

## Usage

To use this module, you need to have OpenEMR installed. Place the module's directory in the `interface/modules/custom_modules/` directory of your OpenEMR installation, and fill any details specified in the in-code comments to your personal settings. It may also be helpful to remove the Notes_in_review_and_development document. Then, you can enable the module from the OpenEMR interface.

A different, or even a custom fine-tuned, LLM model can be substituted for "gpt2". You would need to replace "gpt2" in the llm.py file with the identifier of your desired model, or add in the URL and desired downloader for your fine-tuned model. If your model is hosted on Hugging Face's model hub, the identifier would be in the format "username/model_name".

## Feedback Functionality

Feedback Functionality for the LLM interface may or may not be desired. If implemented they may pose additional security or privacy compliance concerns which will need to be mitigated based on your particular circumstances. This module could be enabled to allow users to provide feedback on the generated responses. While the user interface for feedback submission is partially built-out, the actual implementation of sending and storing feedback requires additional server-side logic, and is currently not operational. This functionality is not tested in this draft.

## Next Steps for Feedback Functionality:

To fully enable feedback functionality, if desired, you will need to develop server-side code to handle the following:

Receiving user feedback submitted via the web interface.
Processing and analyzing the feedback data.
Storing the feedback data for future analysis and potential model improvement.
Mitigate any security or privacy compliance concerns which may arise from feedback functionality.

## Note

This is a high-level overview and the actual implementation may vary depending on the specifics of your OpenEMR installation and the Hugging Face model you want to use. You should refer to the OpenEMR and Hugging Face documentation for more detailed information.

## Disclaimer

Early Work: Use responsibly at your own risk.

Patient data and LLMs which interface with patient data should always be ran locally and encrypted. We have done our best to enable this through having the LLM downloaded and ran on your localy system and through interfacing with patient data within OpenEMR itself. As such, we strongly suggest this be ran on /encrypted/ devices with adequete security and data ethics policies.

This project is still under development and intended for research and educational purposes only. It may not be suitable for clinical use yet. Testers welcome. Implement in your operational systems at your own risk.

While we strive for accuracy, the LLM's responses shouldn't be the sole basis for medical decisions. Always consult a qualified healthcare professional.
