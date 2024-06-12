"use strict"

const displayPanelSelector = '#display-panel'

// const useDynamicDisplayMode = {{use_dynamic_display_mode}}
const useDynamicDisplayMode = true

const observerOptions = {
    "root": null,
    "rootMargin": "0px",
    "threshold": [
        0.0,
        0.5,
        0.75
    ]
}

function handleDisplayUpdate(entries) {
    window.requestAnimationFrame(function(){
        entries.forEach(function(entry){
            let target = entry.target
            // this element has left the screen completely.
            if (entry.intersectionRatio == 0) {
                // do nothing here
            } else if (entry.isIntersecting) {
                let prev = target.previousElementSibling
                console.log("Drawing Intersecting Element", entry.target)
                // hide the entry two slots prior
                if (prev != null) {
                    prev = prev.previousElementSibling
                    if(prev != null) {
                        console.log("Hiding ", prev)
                        prev.style.display = 'none'
                    }
                }
                let next = target.nextElementSibling

                // reveal the next entry coming up
                if (next != null) {
                    console.log("Drawing Next Element", next)
                    next.style.display = 'block'
                }
            }
        })
    })
}


function activateDynamicDisplayMode(scope) {
    let observer = scope.proteinEntryObserver = new IntersectionObserver(
        handleDisplayUpdate, observerOptions)
    document.querySelectorAll(".glycoprotein-entry-detail").forEach(function(entry){
        observer.observe(entry)
    })
}


function hideAllEntries() {
    let detailEntries = document.querySelectorAll(".glycoprotein-entry-detail")
    let n = detailEntries.length
    for(var i = 0; i < n; i++) {
        let entry = detailEntries[i]
        entry.style.display = 'none'
    }
    document.querySelector("#glycoprotein-detail-list").style.display = 'block'
    detailEntries[0].style.display = 'block'
}

function accordionHandler(event) {
    this.classList.toggle("active-accordion");
    console.log(this)
    let panel = this.nextElementSibling;
    if (panel.style.display !== 'none') {
        console.log("Hiding", panel)
        panel.style.display = 'none'
    } else {
        console.log("Showing", panel)
        panel.style.display = 'flex'
    }
}


function debugEntry(event) {
    let structure = this.dataset.structure
    let spectrumContainer = this.querySelector(".annotated-spectrum-container")
    let id = parseInt(this.id.split("-")[2])
    let spectrumId = spectrumContainer ? spectrumContainer.dataset.scanId : null

    console.log({
        structure, id, spectrumId
    })
}


function initViewer(scope) {
    console.log("Initializing Viewer", useDynamicDisplayMode)
    scope.displayPanel = document.querySelector(displayPanelSelector)
    if (useDynamicDisplayMode) {
        activateDynamicDisplayMode(scope)
        let detailEntries = document.querySelectorAll(".glycoprotein-entry-detail")
        detailEntries[0].style.display = 'block'
    }

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

    const glycopeptideRects = querySelectorAll("g.glycopeptide");
    for(let glycopeptide of glycopeptideRects) {
        glycopeptide.addEventListener("mouseover", glycopeptidePileUpMouseOverHandler)
        glycopeptide.addEventListener("mouseout", glycopeptidePileUpMouseOutHandler)
        glycopeptide.addEventListener("click", glycopeptidePileUpMouseClick)
    }

    const glycoproteinTableRows = querySelectorAll("tr.protein-table-row")
    function scrollToGlycoproteinEntry(event) {
        let proteinId = this.dataset.proteinId
        let selector = `#detail-glycoprotein-${proteinId}`
        let element = document.querySelector(selector)
        element.style.display = 'block'
        element.scrollIntoView()
    }

    for(let glycoproteinRow of glycoproteinTableRows) {
        glycoproteinRow.addEventListener("click", scrollToGlycoproteinEntry)
    }

    const glycopeptideTableRows = querySelectorAll("tr.glycopeptide-detail-table-row")
    function scrollToGlycopeptideEntry(event) {
        let glycopeptideId = this.dataset.glycopeptideId
        let selector = `#detail-glycopeptide-${glycopeptideId}`
        document.querySelector(selector).scrollIntoView()
    }

    for(let glycopeptideRow of glycopeptideTableRows) {
        glycopeptideRow.addEventListener("click", scrollToGlycopeptideEntry)
    }

    const accordions = querySelectorAll(".accordion")
    for (let accordion of accordions) {
        accordion.addEventListener("click", accordionHandler)
    }

    const glycopeptideDetailEntries = querySelectorAll(".glycopeptide-detail-container")
    for (let entry of glycopeptideDetailEntries) {
        entry.addEventListener("click", debugEntry)
    }
}


document.addEventListener('DOMContentLoaded', function() {
    initViewer(window)
});


// Taken from https://gist.github.com/Explosion-Scratch/357c2eebd8254f8ea5548b0e6ac7a61b
function compress(string, encoding) {
  const byteArray = new TextEncoder().encode(string);
  const cs = new CompressionStream(encoding);
  const writer = cs.writable.getWriter();
  writer.write(byteArray);
  writer.close();
  return new Response(cs.readable).arrayBuffer();
}


function decompress(byteArray, encoding) {
  const cs = new DecompressionStream(encoding);
  const writer = cs.writable.getWriter();
  writer.write(byteArray);
  writer.close();
  return new Response(cs.readable).arrayBuffer().then(function (arrayBuffer) {
    return new TextDecoder().decode(arrayBuffer);
  });
}