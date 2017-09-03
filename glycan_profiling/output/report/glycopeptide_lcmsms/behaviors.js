"use strict"

const displayPanelSelector = '#display-panel'


function initViewer(scope) {
    scope.displayPanel = document.querySelector(displayPanelSelector)

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
        <div style='margin-bottom: 3px;'>${this.dataset.sequence}</div>
        <div>MS<sub>2</sub> Score: ${this.dataset.ms2Score}</div>
        <div><code>q</code>-Value: ${parseFloat(this.dataset.qValue).toFixed(3)}</div>
        `
        scope.displayPanel.style.display = 'block'
    }

    function glycopeptidePileUpMouseOutHandler(event) {
        scope.displayPanel.style.display = 'none'
    }

    let glycopeptideRects = document.querySelectorAll("g.glycopeptide");
    for(let glycopeptide of glycopeptideRects) {
        glycopeptide.addEventListener("mouseover", glycopeptidePileUpMouseOverHandler)
        glycopeptide.addEventListener("mouseout", glycopeptidePileUpMouseOutHandler)
    }

    let glycoproteinTableRows = document.querySelectorAll("tr.protein-table-row")
    function scrollToGlycoproteinEntry(event) {
        let proteinId = this.dataset.proteinId
        let selector = `#detail-glycoprotein-${proteinId}`
        document.querySelector(selector).scrollIntoView()
    }
    for(let glycoproteinRow of glycoproteinTableRows) {
        glycoproteinRow.addEventListener("click", scrollToGlycoproteinEntry)
    }

    let glycopeptideTableRows = document.querySelectorAll("tr.glycopeptide-detail-table-row")
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
