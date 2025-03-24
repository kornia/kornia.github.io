// Function to format large numbers with K suffix
function formatNumber(num) {
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'k';
    }
    return num.toString();
}

// Function to fetch star count from GitHub API
async function fetchStarCount(repo) {
    try {
        const response = await fetch(`https://api.github.com/repos/${repo}`);
        const data = await response.json();
        return data.stargazers_count;
    } catch (error) {
        console.error(`Error fetching stars for ${repo}:`, error);
        return null;
    }
}

// Function to update star counts
async function updateStarCounts() {
    const starCounts = document.querySelectorAll('.star-count');
    const uniqueRepos = new Set();
    
    // Collect unique repos
    starCounts.forEach(element => {
        uniqueRepos.add(element.dataset.repo);
    });

    // Fetch star counts for each unique repo
    const starData = {};
    for (const repo of uniqueRepos) {
        const count = await fetchStarCount(repo);
        if (count !== null) {
            starData[repo] = count;
        }
    }

    // Update all elements with the fetched data
    starCounts.forEach(element => {
        const repo = element.dataset.repo;
        const count = starData[repo];
        if (count !== undefined) {
            element.querySelector('.count').textContent = formatNumber(count);
        }
    });
}

// Update star counts when the page loads
document.addEventListener('DOMContentLoaded', updateStarCounts); 