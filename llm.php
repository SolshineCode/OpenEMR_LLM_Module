<?php
/**
 * OpenEMR LLM Module - Medical Assistant Interface
 *
 * Provides a web interface for interacting with the LLM backend
 * within the OpenEMR environment.
 *
 * @package   OpenEMR
 * @link      https://www.open-emr.org
 * @author    OpenEMR LLM Module Contributors
 * @license   GNU General Public License v3
 */

// OpenEMR session and access control
// These will be available when running within OpenEMR
if (file_exists("../../globals.php")) {
    require_once("../../globals.php");
    require_once("$srcdir/patient.inc.php");

    use OpenEMR\Common\Acl\AclMain;
    use OpenEMR\Common\Csrf\CsrfUtils;
    use OpenEMR\Core\Header;

    // Check access permissions
    if (!AclMain::aclCheckCore('patients', 'med')) {
        echo xlt('Access denied');
        exit;
    }

    // Get current patient info if in patient context
    $patient_id = isset($pid) ? $pid : null;
    $patient_name = '';
    if ($patient_id) {
        $patient_data = getPatientData($patient_id, "fname,lname");
        if ($patient_data) {
            $patient_name = $patient_data['fname'] . ' ' . $patient_data['lname'];
        }
    }

    $csrf_token = CsrfUtils::collectCsrfToken();
    $in_openemr = true;
} else {
    // Standalone mode for testing
    $patient_id = null;
    $patient_name = '';
    $csrf_token = '';
    $in_openemr = false;

    // Simple translation function for standalone mode
    function xlt($text) { return $text; }
    function xla($text) { return $text; }
    function text($text) { return htmlspecialchars($text, ENT_QUOTES); }
    function addslashes($text) { return addslashes($text); }
}

// Configuration - these should match your Python server settings
$llm_server_url = getenv('LLM_SERVER_URL') ?: 'http://localhost:5000';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo xlt('Medical Assistant LLM'); ?></title>

    <?php if ($in_openemr && class_exists('OpenEMR\Core\Header')): ?>
        <?php Header::setupHeader(['common']); ?>
    <?php endif; ?>

    <style>
        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --success-color: #059669;
            --error-color: #dc2626;
            --warning-color: #d97706;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            background: var(--bg-color);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }

        .llm-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }

        .llm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 10px;
        }

        .llm-header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
        }

        .header-left {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .patient-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            background: #dbeafe;
            color: #1e40af;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .status-indicator.connected {
            background: #dcfce7;
            color: #166534;
        }

        .status-indicator.disconnected {
            background: #fee2e2;
            color: #991b1b;
        }

        .status-indicator.checking {
            background: #fef3c7;
            color: #92400e;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        .chat-container {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: var(--bg-color);
        }

        .message {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.875rem;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: var(--primary-color);
            color: white;
        }

        .message.assistant .message-avatar {
            background: #10b981;
            color: white;
        }

        .message-content {
            flex: 1;
            min-width: 0;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }

        .message-sender {
            font-weight: 600;
            font-size: 0.875rem;
            color: var(--text-primary);
        }

        .message-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .message-body {
            background: var(--card-bg);
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            font-size: 0.9375rem;
            line-height: 1.6;
            color: var(--text-primary);
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .message.assistant .message-body {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        .chat-input-container {
            padding: 16px;
            background: var(--card-bg);
            border-top: 1px solid var(--border-color);
        }

        .input-options {
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }

        .option-checkbox {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.875rem;
            color: var(--text-secondary);
            cursor: pointer;
        }

        .option-checkbox input {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.9375rem;
            resize: none;
            min-height: 48px;
            max-height: 150px;
            font-family: inherit;
            transition: border-color 0.2s;
        }

        .chat-input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .send-button {
            padding: 12px 24px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 0.9375rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .send-button:hover:not(:disabled) {
            background: var(--primary-hover);
        }

        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .send-button .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .feedback-section {
            margin-top: 20px;
            padding: 16px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }

        .feedback-section h3 {
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 12px 0;
            color: var(--text-primary);
        }

        .feedback-buttons {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .feedback-btn {
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }

        .feedback-btn:hover {
            border-color: var(--primary-color);
            background: #eff6ff;
        }

        .feedback-btn.selected {
            border-color: var(--primary-color);
            background: var(--primary-color);
            color: white;
        }

        .feedback-text {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.875rem;
            resize: vertical;
            min-height: 60px;
            font-family: inherit;
        }

        .submit-feedback {
            margin-top: 10px;
            padding: 8px 16px;
            background: var(--success-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
        }

        .submit-feedback:hover {
            background: #047857;
        }

        .disclaimer {
            margin-top: 20px;
            padding: 12px 16px;
            background: #fef3c7;
            border: 1px solid #fcd34d;
            border-radius: 8px;
            font-size: 0.8125rem;
            color: #92400e;
        }

        .disclaimer strong {
            display: block;
            margin-bottom: 4px;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }

        .empty-state svg {
            width: 64px;
            height: 64px;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 8px 12px;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 100% { opacity: 0.3; transform: translateY(0); }
            50% { opacity: 1; transform: translateY(-4px); }
        }

        .error-message {
            padding: 12px 16px;
            background: #fee2e2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            color: #991b1b;
            margin-bottom: 12px;
        }

        .model-info {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        @media (max-width: 600px) {
            .llm-container {
                padding: 10px;
            }

            .input-wrapper {
                flex-direction: column;
            }

            .send-button {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="llm-container">
        <div class="llm-header">
            <div class="header-left">
                <h1><?php echo xlt('Medical Assistant LLM'); ?></h1>
                <?php if ($patient_name): ?>
                    <span class="patient-badge">
                        <?php echo xlt('Patient'); ?>: <?php echo text($patient_name); ?>
                    </span>
                <?php endif; ?>
            </div>
            <div id="server-status" class="status-indicator checking">
                <span class="status-dot"></span>
                <span class="status-text"><?php echo xlt('Checking'); ?>...</span>
            </div>
        </div>

        <div class="chat-container">
            <div id="chat-messages" class="chat-messages">
                <div class="empty-state" id="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <p><?php echo xlt('Ask a medical question to get started'); ?></p>
                    <p class="model-info" id="model-info"></p>
                </div>
            </div>

            <div class="chat-input-container">
                <div id="error-container"></div>

                <div class="input-options">
                    <?php if ($patient_id): ?>
                        <label class="option-checkbox">
                            <input type="checkbox" id="include-patient-data">
                            <?php echo xlt('Include patient context'); ?>
                        </label>
                    <?php endif; ?>
                </div>

                <div class="input-wrapper">
                    <textarea
                        id="prompt-input"
                        class="chat-input"
                        placeholder="<?php echo xla('Type your medical question here...'); ?>"
                        rows="1"
                    ></textarea>
                    <button id="send-btn" class="send-button">
                        <span class="btn-text"><?php echo xlt('Send'); ?></span>
                    </button>
                </div>
            </div>
        </div>

        <div class="feedback-section" id="feedback-section" style="display: none;">
            <h3><?php echo xlt('Was this response helpful?'); ?></h3>
            <div class="feedback-buttons">
                <button class="feedback-btn" data-rating="helpful"><?php echo xlt('Helpful'); ?></button>
                <button class="feedback-btn" data-rating="somewhat"><?php echo xlt('Somewhat'); ?></button>
                <button class="feedback-btn" data-rating="not-helpful"><?php echo xlt('Not Helpful'); ?></button>
            </div>
            <textarea
                id="feedback-text"
                class="feedback-text"
                placeholder="<?php echo xla('Additional feedback (optional)'); ?>"
            ></textarea>
            <button id="submit-feedback" class="submit-feedback"><?php echo xlt('Submit Feedback'); ?></button>
        </div>

        <div class="disclaimer">
            <strong><?php echo xlt('Important Notice'); ?></strong>
            <?php echo xlt('This AI assistant is for informational purposes only and should not replace professional medical judgment. Always verify information and consult appropriate healthcare providers for clinical decisions.'); ?>
        </div>
    </div>

    <script>
        (function() {
            'use strict';

            const LLM_SERVER_URL = '<?php echo addslashes($llm_server_url); ?>';
            const PATIENT_ID = <?php echo $patient_id ? "'" . addslashes($patient_id) . "'" : 'null'; ?>;

            let currentRequestId = null;
            let isLoading = false;

            // DOM Elements
            const chatMessages = document.getElementById('chat-messages');
            const promptInput = document.getElementById('prompt-input');
            const sendBtn = document.getElementById('send-btn');
            const serverStatus = document.getElementById('server-status');
            const emptyState = document.getElementById('empty-state');
            const errorContainer = document.getElementById('error-container');
            const feedbackSection = document.getElementById('feedback-section');
            const includePatientData = document.getElementById('include-patient-data');
            const modelInfo = document.getElementById('model-info');

            // Check server health and get config
            async function checkServerHealth() {
                try {
                    const response = await fetch(LLM_SERVER_URL + '/health', {
                        method: 'GET'
                    });

                    if (response.ok) {
                        const data = await response.json();
                        setServerStatus('connected', '<?php echo xla("Connected"); ?>');

                        // Get and display model info
                        if (modelInfo) {
                            modelInfo.textContent = 'Backend: ' + (data.backend || 'unknown');
                        }
                        return true;
                    }
                } catch (e) {
                    console.error('Health check failed:', e);
                }

                setServerStatus('disconnected', '<?php echo xla("Disconnected"); ?>');
                return false;
            }

            function setServerStatus(status, text) {
                serverStatus.className = 'status-indicator ' + status;
                serverStatus.querySelector('.status-text').textContent = text;
            }

            // Auto-resize textarea
            promptInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 150) + 'px';
            });

            // Handle Enter key
            promptInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Send button click
            sendBtn.addEventListener('click', sendMessage);

            async function sendMessage() {
                const prompt = promptInput.value.trim();
                if (!prompt || isLoading) return;

                // Clear error
                errorContainer.innerHTML = '';

                // Hide empty state
                if (emptyState) {
                    emptyState.style.display = 'none';
                }

                // Add user message
                addMessage('user', prompt);

                // Clear input
                promptInput.value = '';
                promptInput.style.height = 'auto';

                // Show loading
                setLoading(true);
                const loadingEl = addTypingIndicator();

                try {
                    const payload = {
                        prompt: prompt,
                        patient_id: PATIENT_ID,
                        include_patient_data: includePatientData && includePatientData.checked
                    };

                    const response = await fetch(LLM_SERVER_URL + '/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(payload)
                    });

                    // Remove loading indicator
                    loadingEl.remove();

                    if (!response.ok) {
                        const errorData = await response.json().catch(function() { return {}; });
                        throw new Error(errorData.error || 'Server error: ' + response.status);
                    }

                    const data = await response.json();
                    currentRequestId = data.request_id;

                    // Add assistant message
                    addMessage('assistant', data.response, data.model);

                    // Show feedback section
                    feedbackSection.style.display = 'block';

                } catch (error) {
                    loadingEl.remove();
                    showError(error.message);
                    console.error('Generation failed:', error);
                } finally {
                    setLoading(false);
                }
            }

            function addMessage(role, content, model) {
                const messageEl = document.createElement('div');
                messageEl.className = 'message ' + role;

                const avatar = role === 'user' ? 'U' : 'AI';
                const sender = role === 'user' ?
                    '<?php echo xla("You"); ?>' :
                    '<?php echo xla("Medical Assistant"); ?>';
                const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

                let modelBadge = '';
                if (model && role === 'assistant') {
                    modelBadge = '<span class="model-info">(' + escapeHtml(model) + ')</span>';
                }

                messageEl.innerHTML =
                    '<div class="message-avatar">' + avatar + '</div>' +
                    '<div class="message-content">' +
                        '<div class="message-header">' +
                            '<span class="message-sender">' + sender + '</span>' +
                            '<span class="message-time">' + time + '</span>' +
                            modelBadge +
                        '</div>' +
                        '<div class="message-body">' + escapeHtml(content) + '</div>' +
                    '</div>';

                chatMessages.appendChild(messageEl);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function addTypingIndicator() {
                const el = document.createElement('div');
                el.className = 'message assistant';
                el.innerHTML =
                    '<div class="message-avatar">AI</div>' +
                    '<div class="message-content">' +
                        '<div class="message-body">' +
                            '<div class="typing-indicator">' +
                                '<span></span>' +
                                '<span></span>' +
                                '<span></span>' +
                            '</div>' +
                        '</div>' +
                    '</div>';
                chatMessages.appendChild(el);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                return el;
            }

            function setLoading(loading) {
                isLoading = loading;
                sendBtn.disabled = loading;
                if (loading) {
                    sendBtn.innerHTML = '<span class="spinner"></span>';
                } else {
                    sendBtn.innerHTML = '<span class="btn-text"><?php echo xlt("Send"); ?></span>';
                }
            }

            function showError(message) {
                errorContainer.innerHTML =
                    '<div class="error-message">' +
                        '<strong><?php echo xla("Error"); ?>:</strong> ' + escapeHtml(message) +
                    '</div>';
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            // Feedback handling
            const feedbackBtns = document.querySelectorAll('.feedback-btn');
            let selectedRating = null;

            feedbackBtns.forEach(function(btn) {
                btn.addEventListener('click', function() {
                    feedbackBtns.forEach(function(b) { b.classList.remove('selected'); });
                    this.classList.add('selected');
                    selectedRating = this.dataset.rating;
                });
            });

            document.getElementById('submit-feedback').addEventListener('click', async function() {
                if (!selectedRating || !currentRequestId) return;

                const feedbackText = document.getElementById('feedback-text').value;

                try {
                    await fetch(LLM_SERVER_URL + '/feedback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            request_id: currentRequestId,
                            rating: selectedRating,
                            feedback_text: feedbackText,
                            helpful: selectedRating === 'helpful'
                        })
                    });

                    // Reset feedback UI
                    feedbackSection.style.display = 'none';
                    feedbackBtns.forEach(function(b) { b.classList.remove('selected'); });
                    document.getElementById('feedback-text').value = '';
                    selectedRating = null;

                    // Show thank you message
                    addMessage('assistant', '<?php echo xla("Thank you for your feedback!"); ?>');

                } catch (error) {
                    console.error('Failed to submit feedback:', error);
                }
            });

            // Initial health check
            checkServerHealth();

            // Periodic health check
            setInterval(checkServerHealth, 30000);

        })();
    </script>
</body>
</html>
