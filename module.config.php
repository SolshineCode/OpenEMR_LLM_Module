<?php

/**
 * OpenEMR LLM Module Configuration
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    OpenEMR LLM Module Contributors
 * @license   GNU General Public License v3
 */

return [
    'name' => 'Medical Assistant LLM',
    'description' => 'AI-powered medical assistant integrating local LLM inference (llama.cpp, Ollama) with OpenEMR for patient data-aware clinical decision support.',
    'version' => '2.0',
    'author' => 'OpenEMR LLM Module Contributors',
    'email' => 'caleb.deleeuw.polychora@gmail.com',
    'license' => 'GPL-3.0',

    // Module routes
    'routes' => [
        'llm' => [
            'type' => 'Laminas\Router\Http\Literal',
            'options' => [
                'route' => '/interface/modules/custom_modules/llm/llm.php',
                'defaults' => [
                    'controller' => 'LLM',
                    'action' => 'index',
                ],
            ],
        ],
    ],

    // Module menu entry
    'menu' => [
        'label' => 'Medical Assistant LLM',
        'menu_id' => 'llm0',
        'target' => 'mod',
        'url' => '/interface/modules/custom_modules/llm/llm.php',
        'hook' => 'custom_modules.llm0',
        'acl' => ['admin', 'super'],
    ],

    // Module dependencies
    'dependencies' => [
        'openemr' => '>=7.0.0',
    ],
];
