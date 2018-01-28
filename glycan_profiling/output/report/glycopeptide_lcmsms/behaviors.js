"use strict"

const displayPanelSelector = '#display-panel'


function initViewer(scope) {
    console.log("Initializing Viewer")
    scope.displayPanel = document.querySelector(displayPanelSelector)

    function querySelectorAll(selector) {
        return Array.from(document.querySelectorAll(selector))
    }

    function glycopeptidePileUpMouseOverHandler(event) {
        const leftSideThreshold = window.screen.width / 2
        let isLeftSide = event.screenX < leftSideThreshold
        // show popup on the opposite side
        if(!isLeftSide) {
            scope.displayPanel.style.left = "25px"
        } else {
            scope.displayPanel.style.left = `${leftSideThreshold + 25}px`
        }
        scope.displayPanel.innerHTML = `
        <div style='margin-bottom: 6px;'>${this.dataset.sequence}</div>
        <div>MS<sub>2</sub> Score: ${this.dataset.ms2Score}</div>
        <div><code>q</code>-Value: ${parseFloat(this.dataset.qValue).toFixed(3)}</div>
        <div># Spectrum Maches: ${this.dataset.spectraCount}</div>
        `
        scope.displayPanel.style.display = 'block'
    }

    function glycopeptidePileUpMouseOutHandler(event) {
        scope.displayPanel.style.display = 'none'
    }

    function glycopeptidePileUpMouseClick(event) {
        let glycopeptideId = this.dataset.recordId
        let selector = `#detail-glycopeptide-${glycopeptideId}`
        console.log(`Looking for #detail-glycopeptide-${glycopeptideId}`)
        document.querySelector(selector).scrollIntoView()
    }

    let glycopeptideRects = querySelectorAll("g.glycopeptide");
    for(let glycopeptide of glycopeptideRects) {
        glycopeptide.addEventListener("mouseover", glycopeptidePileUpMouseOverHandler)
        glycopeptide.addEventListener("mouseout", glycopeptidePileUpMouseOutHandler)
        glycopeptide.addEventListener("click", glycopeptidePileUpMouseClick)
    }

    let glycoproteinTableRows = querySelectorAll("tr.protein-table-row")
    function scrollToGlycoproteinEntry(event) {
        let proteinId = this.dataset.proteinId
        let selector = `#detail-glycoprotein-${proteinId}`
        document.querySelector(selector).scrollIntoView()
    }

    for(let glycoproteinRow of glycoproteinTableRows) {
        glycoproteinRow.addEventListener("click", scrollToGlycoproteinEntry)
    }

    let glycopeptideTableRows = querySelectorAll("tr.glycopeptide-detail-table-row")
    function scrollToGlycopeptideEntry(event) {
        let glycopeptideId = this.dataset.glycopeptideId
        let selector = `#detail-glycopeptide-${glycopeptideId}`
        document.querySelector(selector).scrollIntoView()
    }

    for(let glycopeptideRow of glycopeptideTableRows) {
        glycopeptideRow.addEventListener("click", scrollToGlycopeptideEntry)
    }
}


document.addEventListener('DOMContentLoaded', function() {
    initViewer(window)
});
