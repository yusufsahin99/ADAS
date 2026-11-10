// Copy BibTeX to clipboard
function copyBibTeX() {
  var code = document.getElementById('bibtex-code');
  var button = document.querySelector('.copy-bibtex-btn');
  var label = button.querySelector('.copy-text');
  if (!code) return;

  function done() {
    button.classList.add('copied');
    label.textContent = 'Cop';
    setTimeout(function () {
      button.classList.remove('copied');
      label.textContent = 'Copy';
    }, 2000);
  }

  function selectText() {
    var range = document.createRange();
    range.selectNodeContents(code);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    label.textContent = 'Selected';
  }

  try {
    navigator.clipboard.writeText(code.textContent).then(done, selectText);
  } catch (e) {
    selectText();
  }
}

// Scroll to top
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  var btn = document.querySelector('.scroll-to-top');
  if (!btn) return;
  if (window.pageYOffset > 300) btn.classList.add('visible');
  else btn.classList.remove('visible');
});
