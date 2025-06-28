document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("news_text");
    const submitBtn = document.getElementById("submit_btn");
    const wordCountDisplay = document.getElementById("word_count_display");
    const MIN_WORDS = 20;

    function updateWordCount() {
        const text = textarea.value.trim();
        const words = text.split(/\s+/).filter(word => word.length > 0);
        const count = words.length;

        wordCountDisplay.textContent = `Word count: ${count}`;
        submitBtn.disabled = count < MIN_WORDS;
    }

    textarea.addEventListener("input", updateWordCount);
});
