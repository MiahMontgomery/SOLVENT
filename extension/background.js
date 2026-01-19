// SOLVENT Background Service Worker
// Handles browser automation and agent logic

const SOFT_GREEN = '#C5F5D5';
let activeSession = null;
let budget = { limit: 100, spent: 0 };

// Session state
class SolventSession {
    constructor() {
          this.accounts = [];
          this.tasks = [];
          this.status = 'idle';
          this.thoughts = [];
    }

  addThought(text) {
        const thought = {
                text,
                timestamp: Date.now()
        };
        this.thoughts.push(thought);
        this.broadcastThought(thought);
  }

  broadcastThought(thought) {
        chrome.runtime.sendMessage({
                type: 'agent_thought',
                data: thought
        }).catch(() => {}); // Popup might not be open
  }

  async scanGmail() {
        this.status = 'scanning';
        this.addThought('navigating to myactivity.google.com...');

      try {
              const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

          await chrome.tabs.update(tab.id, {
                    url: 'https://myactivity.google.com'
          });

          // Wait for page load
          await new Promise(resolve => setTimeout(resolve, 2000));

          this.addThought('analyzing google activity data...');

          // Inject scraper
          await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: analyzeGoogleActivity
          });

          this.addThought('scan complete. building checklist...');

      } catch (error) {
              this.addThought(`error: ${error.message}`);
      }
  }

  async scanReddit() {
        this.addThought('navigating to reddit user profile...');

      try {
              const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

          await chrome.tabs.update(tab.id, {
                    url: 'https://www.reddit.com/user/me/posts'
          });

          await new Promise(resolve => setTimeout(resolve, 2000));

          this.addThought('analyzing reddit post history...');

          await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: analyzeRedditContent
          });

          this.addThought('reddit scan complete.');

      } catch (error) {
              this.addThought(`error: ${error.message}`);
      }
  }

  async dissolveGoogleActivity() {
        this.addThought('initiating google activity dissolution...');
        this.status = 'dissolving';

      try {
              const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

          this.addThought('locating delete controls...');

          await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: deleteGoogleActivity
          });

          this.addThought('deletion initiated. monitoring progress...');

      } catch (error) {
              this.addThought(`error: ${error.message}`);
      }
  }
}

// Content script functions (injected into pages)
function analyzeGoogleActivity() {
    // Find activity metrics on the page
  const metrics = {
        searches: 0,
        locations: 0,
        youtube: 0
  };

  // This would parse the actual page content
  // For now, simulated detection
  const activityCards = document.querySelectorAll('[role="article"]');
    metrics.searches = activityCards.length;

  chrome.runtime.sendMessage({
        type: 'activity_found',
        data: metrics
  });
}

function analyzeRedditContent() {
    const posts = document.querySelectorAll('[data-testid="post-container"]');
    const comments = document.querySelectorAll('[data-testid="comment"]');

  chrome.runtime.sendMessage({
        type: 'reddit_found',
        data: {
                posts: posts.length,
                comments: comments.length
        }
  });
}

function deleteGoogleActivity() {
    // Navigate to delete controls
  const deleteButton = document.querySelector('button[aria-label*="Delete"]');

  if (deleteButton) {
        deleteButton.click();

      // Wait for modal
      setTimeout(() => {
              const allTimeOption = Array.from(document.querySelectorAll('span'))
                .find(el => el.textContent.includes('All time'));

                       if (allTimeOption) {
                                 allTimeOption.click();

                setTimeout(() => {
                            const confirmButton = document.querySelector('button[aria-label*="Delete"]');
                            if (confirmButton) {
                                          confirmButton.click();
                                          chrome.runtime.sendMessage({ type: 'deletion_complete' });
                            }
                }, 500);
                       }
      }, 500);
  } else {
        chrome.runtime.sendMessage({ 
                                         type: 'deletion_error',
                message: 'delete button not found. manual login may be required.'
        });
  }
}

// Message handlers
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
      case 'start_session':
              activeSession = new SolventSession();
              budget = message.budget || { limit: 100, spent: 0 };
              sendResponse({ status: 'session_started' });
              break;

      case 'scan_gmail':
              if (!activeSession) {
                        activeSession = new SolventSession();
              }
              activeSession.scanGmail();
              sendResponse({ status: 'scanning' });
              break;

      case 'scan_reddit':
              if (!activeSession) {
                        activeSession = new SolventSession();
              }
              activeSession.scanReddit();
              sendResponse({ status: 'scanning' });
              break;

      case 'dissolve_google':
              if (activeSession) {
                        activeSession.dissolveGoogleActivity();
                        sendResponse({ status: 'dissolving' });
              }
              break;

      case 'get_session':
              sendResponse({ 
                                   session: activeSession,
                        budget: budget
              });
              break;

      case 'activity_found':
      case 'reddit_found':
              if (activeSession) {
                        activeSession.accounts.push(message.data);
              }
              break;

      case 'deletion_complete':
              if (activeSession) {
                        activeSession.addThought('dissolution complete. return to blank.');
                        activeSession.status = 'complete';
              }
              break;
    }

                                       return true; // Keep channel open for async responses
});

// Budget tracking
function trackSpend(amount) {
    budget.spent += amount;

  if (budget.spent >= budget.limit) {
        if (activeSession) {
                activeSession.addThought(`budget limit reached ($${budget.limit}). session terminated.`);
                activeSession.status = 'budget_exceeded';
        }
        return false; // Stop execution
  }

  return true; // Continue
}

// Initialize
chrome.runtime.onInstalled.addListener(() => {
    console.log('SOLVENT installed. return to blank.');
});
