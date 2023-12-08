<?php

return [
    'name' => 'Medical Assistant LLM',
    'description' => 'A module that integrates a configurable Hugging Face language model for medical assistance, including patient data access and feedback functionality.',
    'version' => '1.1',
    'author' => 'Caleb DeLeeuw',
    'email' => 'caleb.deleeuw.polychora@gmail.com',
    'routes' => [
        'llm/response' => [
            'type' => 'Zend\Mvc\Router\Http\Literal',
            'options' => [
                'route' => '/response.php',
            ],
        ],
    ],
];
