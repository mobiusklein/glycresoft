"use strict"

function initViewer(scope) {
    let glycanTableRows = document.querySelectorAll("tr.glycan-detail-table-row")
    function scrollToGlycanEntry(event) {
        let glycanId = this.dataset.glycanId
        let selector = `#detail-${glycanId}`
        document.querySelector(selector).scrollIntoView()
    }
    for(let glycanRow of glycanTableRows) {
        glycanRow.addEventListener("click", scrollToGlycanEntry)
    }
}

document.addEventListener('DOMContentLoaded', function() {
    initViewer(window)
});
