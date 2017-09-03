"use strict"

const displayPanelSelector = '#display-panel'


function initViewer(scope) {
    scope.displayPanel = document.querySelector(displayPanelSelector)

    function glycopeptidePileUpMouseOverHandler(event) {
        console.log(event)
        const leftSideThreshold = window.screen.width / 2
        let isLeftSide = event.screenX < leftSideThreshold
        // show popup on the opposite side
        if(!isLeftSide) {
            scope.displayPanel.style.left = "25px"
        } else {
            scope.displayPanel.style.left = `${leftSideThreshold + 25}px`
        }
        scope.displayPanel.innerHTML = `<div style='margin-bottom: 3px;'>${this.dataset.sequence}</div>
        MS<sub>2</sub> Score: ${this.dataset.ms2Score}<br>
        <code>q</code>-Value: ${parseFloat(this.dataset.qValue).toFixed(3)}<br>
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
}


document.addEventListener('DOMContentLoaded', function() {
    initViewer(window)
    console.log("loaded")
});
