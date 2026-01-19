// SOLVENT Popup UI Controller

let isInside = false;
let sessionActive = false;

const elements = {
    bubble: document.getElementById('bubble'),
    bubbleContainer: document.getElementById('bubble-container'),
    interface: document.getElementById('interface'),
    thoughts: document.getElementById('thoughts'),
    input: document.getElementById('command-input'),
    budgetDisplay: document.getElementById('budget-display'),
    spent: document.getElementById('spent'),
    limit: document.getElementById('limit')
};

// Bubble click - "enter" the bubble
elements.bubble.addEventListener('click', () => {
    if (!isInside) {
          enterBubble();
    }
});

function enterBubble() {
    isInside = true;

  // Hide bubble, show interface
  elements.bubbleContainer.style.display = 'none';
    elements.interface.classList.add('active');
    elements.budgetDisplay.classList.add('active');

  // Start session
  chrome.runtime.sendMessage({ type: 'start_session', budget: { limit: 100, spent: 0 } });

  addThought('agent initialized. ready for commands.');

  elements.input.focus();
}

// Command input
elements.input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
          const command = elements.input.value.trim().toLowerCase();
          if (command) {
                  processCommand(command);
                  elements.input.value = '';
          }
    }
});

function processCommand(command) {
    addThought(`$ ${command}`, 'user');

  if (command.includes('gmail') || command.includes('google')) {
        chrome.runtime.sendMessage({ type: 'scan_gmail' });
  } 
  else if (command.includes('reddit')) {
        chrome.runtime.sendMessage({ type: 'scan_reddit' });
  }
    else if (command.includes('dissolve') && command.includes('google')) {
          chrome.runtime.sendMessage({ type: 'dissolve_google' });
    }
    else if (command.includes('budget')) {
          const match = command.match(/\$?(\d+)/);
          if (match) {
                  const limit = parseInt(match[1]);
                  elements.limit.textContent = limit;
                  addThought(`budget limit set to $${limit}`);
          }
    }
    else {
          addThought(`processing: ${command}`);
          setTimeout(() => {
                  addThought('command acknowledged. specify platform to scan.');
          }, 500);
    }
}

function addThought(text, type = 'agent') {
    const thought = document.createElement('div');
    thought.className = 'thought';
    thought.style.color = type === 'user' ? 'rgba(255,255,255,0.6)' : '#C5F5D5';
    thought.textContent = text;

  elements.thoughts.appendChild(thought);
    elements.thoughts.scrollTop = elements.thoughts.scrollHeight;
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
      case 'agent_thought':
              addThought(message.data.text);
              break;

      case 'activity_found':
              const data = message.data;
              addThought(`found: ${data.searches || 0} searches, ${data.locations || 0} locations`);
              break;

      case 'reddit_found':
              addThought(`found: ${message.data.posts} posts, ${message.data.comments} comments`);
              break;

      case 'deletion_complete':
              addThought('dissolution complete. return to blank.');
              break;

      case 'deletion_error':
              addThought(message.message);
              break;
    }
});

// Budget updates
function updateBudget(spent, limit) {
    elements.spent.textContent = spent.toFixed(2);
    elements.limit.textContent = limit;

  const percentage = (spent / limit) * 100;
    if (percentage > 90) {
          elements.budgetDisplay.style.color = '#ff6b6b';
    } else if (percentage > 70) {
          elements.budgetDisplay.style.color = '#ffd93d';
    }
}

// Restore session on popup open
chrome.runtime.sendMessage({ type: 'get_session' }, (response) => {
    if (response && response.session && response.session.status !== 'idle') {
          // Resume session
      enterBubble();

      // Restore thoughts
      if (response.session.thoughts) {
              response.session.thoughts.forEach(thought => {
                        addThought(thought.text);
              });
      }

      // Update budget
      if (response.budget) {
              updateBudget(response.budget.spent, response.budget.limit);
      }
    }
});
