# OpenEMR_LLM_Module
An OpenEMR module creating a new tab with an LLM downloaded from Hugging Face and ran locally (to assure patient data privacy compliance) and able to analyze patient data conversationally.

## Medical Assistant LLM Module for OpenEMR

This project is a custom module for OpenEMR that integrates a Hugging Face language model to assist with medical tasks. The module adds a new tab to the OpenEMR interface, where users can interact with the language model.

## Features

- Integration with a Hugging Face language model: The module uses the `transformers` library to download and use a language model. The model can be loaded with `AutoModelForCausalLM.from_pretrained(model_name)`.
- New tab in OpenEMR: The module adds a new tab to the OpenEMR interface, where users can interact with the language model. The tab includes a prompt window where users can input a prompt, and an answer window where the model's response is displayed.
- Connection to OpenEMR patient data: The module can retrieve patient data from OpenEMR and pass it to the language model.

## Code Overview

The module consists of several parts:

- `module.info`: This file describes the module and is required by OpenEMR.
- `llm.py`: This Python script uses the `transformers` library to download and use a language model. It defines a function `generate_response` that takes a prompt as input, encodes it into a format that the model can understand, generates a response, and decodes the response into text.
- `menu_data.json`: This file defines a new menu item that links to the module's GUI.
- `llm.php`: This PHP file serves the module's GUI. It includes a form for the user to input a prompt, and a display area to show the model's response. It uses AJAX to send the prompt to a PHP script that interacts with the model and returns the response.

## Usage

To use this module, you need to have OpenEMR installed. Place the module's directory in the `interface/modules/custom_modules/` directory of your OpenEMR installation. Then, you can enable the module from the OpenEMR interface.

## Feedback Functionality

This module allows users to provide feedback on the generated responses. While the user interface for feedback submission is functional, the actual implementation of sending and storing feedback requires additional server-side logic. This functionality is not tested in this draft.

## Next Steps for Feedback Functionality:

To fully enable feedback functionality, you will need to develop server-side code to handle the following:

Receiving user feedback submitted via the web interface.
Processing and analyzing the feedback data.
Storing the feedback data for future analysis and potential model improvement.

## Note

This is a high-level overview and the actual implementation may vary depending on the specifics of your OpenEMR installation and the Hugging Face model you want to use. You should refer to the OpenEMR and Hugging Face documentation for more detailed information.

## Disclaimer

Early Work: Use responsibly at your own risk.

This project is still under development and intended for research and educational purposes only. It may not be suitable for clinical use yet.

While we strive for accuracy, the LLM's responses shouldn't be the sole basis for medical decisions. Always consult a qualified healthcare professional.
