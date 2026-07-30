const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Build evidence panel from structured_facts array
function createEvidencePanel(facts) {
    if (!facts || facts.length === 0) return null;

    const panel = document.createElement('details');
    panel.classList.add('evidence-panel');

    const title = document.createElement('summary');
    title.classList.add('evidence-title');
    title.innerHTML = '📊 <span>View Evidence from Knowledge Graph</span>';
    panel.appendChild(title);

    const grid = document.createElement('div');
    grid.classList.add('evidence-grid');

    facts.forEach(fact => {
        const card = document.createElement('div');
        card.classList.add('evidence-card');

        // Try to extract useful keys from Cypher result
        const entries = Object.entries(fact).filter(([k, v]) => v !== null && v !== undefined);
        entries.forEach(([key, value]) => {
            const row = document.createElement('div');
            row.classList.add('evidence-row');
            const label = key.replace(/^[pr]\.|_/g, ' ').trim();
            row.innerHTML = `<span class="ev-key">${label}</span><span class="ev-val">${value}</span>`;
            card.appendChild(row);
        });

        if (card.children.length > 0) {
            grid.appendChild(card);
        }
    });

    if (grid.children.length === 0) return null;
    panel.appendChild(grid);
    return panel;
}

function appendMessage(sender, text, evidence = []) {
    const welcomeHero = document.getElementById('welcome-hero');
    if (welcomeHero) welcomeHero.remove();

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.innerText = sender === 'user' ? 'U' : 'AI';

    const wrapper = document.createElement('div');
    wrapper.classList.add('message-wrapper');

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');

    // Strip out the bracketed evidence tags from the display text
    let cleanText = text.replace(/\[\[.*?\]\]/g, '').trim();

    if (sender === 'system') {
        // Configure marked.js for rich GitHub-flavored markdown with line breaks
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true
            });
        }

        // Convert Unicode bullets (•) and irregular asterisks into standard Markdown list items
        cleanText = cleanText.replace(/^[ \t]*•[ \t]*/gm, '- ');
        cleanText = cleanText.replace(/^[ \t]*\*[ \t]+/gm, '- ');

        // Automatically highlight/bold place names right before an em-dash (— or -- or -)
        // Only bold if it looks like a proper noun (starts with uppercase, has 2+ words)
        cleanText = cleanText.replace(/^- ([A-Z][^—\-\n]{2,44}) (—|--) /gm, '- **$1** — ');

        bubble.innerHTML = typeof marked !== 'undefined' ? marked.parse(cleanText) : cleanText;
    } else {
        bubble.innerText = cleanText;
    }
    wrapper.appendChild(bubble);

    // Attach evidence panel for AI messages
    if (sender === 'system' && evidence && evidence.length > 0) {
        const panel = createEvidencePanel(evidence);
        if (panel) wrapper.appendChild(panel);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(wrapper);

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';

    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'system');
    loadingDiv.innerHTML = `<div class="avatar">AI</div><div class="message-wrapper"><div class="bubble loading">Thinking<span class="dots"><span>.</span><span>.</span><span>.</span></span></div></div>`;
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });

        const data = await response.json();
        chatContainer.removeChild(loadingDiv);
        appendMessage('system', data.response, data.evidence || []);

        if (data.is_itinerary_request) {
            window.location.href = 'planner.html';
        }
    } catch (error) {
        console.error('Error:', error);
        if (chatContainer.contains(loadingDiv)) chatContainer.removeChild(loadingDiv);
        appendMessage('system', 'Sorry, I encountered an error. Please try again.');
    }
}


// The composer is a real <form>, so Enter and the send button both route
// through submit — preventDefault stops the browser navigating away.
const chatForm = document.getElementById('chat-form');
if (chatForm) {
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
    });
} else {
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}

// Fill the box from a suggestion card and send it.
function askSuggestion(text) {
    userInput.value = text;
    sendMessage();
}

// Deep link from the Explore page: chat.html?q=Tell%20me%20about...
document.addEventListener('DOMContentLoaded', () => {
    const q = new URLSearchParams(window.location.search).get('q');
    if (q) askSuggestion(q);
});

