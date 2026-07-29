let globalItineraryMap = null;

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('trip-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const payload = {
            start_place: formData.get('start_place'),
            days: parseInt(formData.get('days')),
            hours_per_day: parseFloat(formData.get('hours_per_day')),
            max_places: parseInt(formData.get('max_places')),
            must_visit: formData.get('must_visit'),
            attraction_priority: parseInt(formData.get('attraction_priority')),
            budget_priority: parseInt(formData.get('budget_priority')),
            time_priority: parseInt(formData.get('time_priority')),
            popular_priority: parseInt(formData.get('popular_priority')),
            question: "Plan a trip for me"
        };

        const submitBtn = document.getElementById('generate-btn');
        submitBtn.disabled = true;
        submitBtn.innerText = 'Generating...';
        
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.style.display = 'flex';
        
        // Don't hide planner-main, just let the overlay cover it
        const plannerPlaceholder = document.getElementById('planner-placeholder');
        if (plannerPlaceholder) plannerPlaceholder.style.display = 'none';

        try {
            const response = await fetch('/api/plan_itinerary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            loadingOverlay.style.display = 'none';
            submitBtn.disabled = false;
            submitBtn.innerText = 'Generate Itinerary';

            if (data.moip_result) {
                renderItineraryWidget(data.response, data.moip_result);
            } else {
                alert("Sorry, could not generate itinerary.");
            }
        } catch (error) {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none';
            submitBtn.disabled = false;
            submitBtn.innerText = 'Generate Itinerary';
            alert('Sorry, I encountered an error while generating the itinerary. Please try again.');
        }
    });
});

function renderItineraryWidget(llmSummary, moipResult) {
    const itineraryWidget = document.getElementById('itinerary-widget');
    itineraryWidget.style.display = 'flex';
    
    const plannerPlaceholder = document.getElementById('planner-placeholder');
    if (plannerPlaceholder) plannerPlaceholder.style.display = 'none';
    
    // Set LLM Summary
    const summaryDiv = document.getElementById('itinerary-summary');
    summaryDiv.innerHTML = marked.parse(llmSummary);

    // Build Timeline
    const timelineDiv = document.getElementById('itinerary-timeline');
    timelineDiv.innerHTML = ''; // Clear previous
    
    const itinerary = moipResult.itinerary || [];
    const allCoords = [];
    
    itinerary.forEach(day => {
        const dayCard = document.createElement('div');
        dayCard.className = 'day-card';
        
        const dayTitle = document.createElement('h4');
        dayTitle.innerText = `Day ${day.day} (Est. ${Math.round(day.time_min / 60)} hrs)`;
        dayCard.appendChild(dayTitle);
        
        if (day.detailed_places) {
            day.detailed_places.forEach((place, index) => {
                const pItem = document.createElement('div');
                pItem.className = 'place-item';
                
                let html = `<strong>${index + 1}. ${place.name}</strong>`;
                if (place.description) {
                    html += `<div class="place-desc">${place.description}</div>`;
                }
                
                let pillsHtml = '';
                if (place.features) {
                    place.features.forEach(f => pillsHtml += `<span class="pill">${f}</span>`);
                }
                if (place.activities) {
                    place.activities.forEach(a => pillsHtml += `<span class="pill activity">${a}</span>`);
                }
                if (pillsHtml) {
                    html += `<div class="pill-container">${pillsHtml}</div>`;
                }
                
                pItem.innerHTML = html;
                dayCard.appendChild(pItem);
                

                
                if (place.lat && place.lon) {
                    allCoords.push([place.lat, place.lon]);
                    // Add hover effect
                    pItem.addEventListener('mouseenter', () => {
                        if (globalItineraryMap) {
                            globalItineraryMap.setView([place.lat, place.lon], 13);
                        }
                    });
                }
            });
        }
        timelineDiv.appendChild(dayCard);
    });

    // Initialize Map
    setTimeout(() => {
        const mapContainer = document.getElementById('itinerary-map');
        if (globalItineraryMap) {
            globalItineraryMap.remove();
        }
        
        globalItineraryMap = L.map(mapContainer);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(globalItineraryMap);
        
        if (allCoords.length > 0) {
            globalItineraryMap.fitBounds(L.latLngBounds(allCoords));
            
            // Draw Real Road Route via OSRM API
            async function drawRoute(coords) {
                if (coords.length < 2) return;
                const coordStr = coords.map(c => `${c[1]},${c[0]}`).join(';');
                const url = `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`;
                try {
                    const response = await fetch(url);
                    const data = await response.json();
                    if (data.routes && data.routes.length > 0) {
                        const geojson = data.routes[0].geometry;
                        L.geoJSON(geojson, {
                            style: { color: '#2563eb', weight: 5, opacity: 0.85 }
                        }).addTo(globalItineraryMap);
                    } else {
                        // Fallback to straight lines
                        L.polyline(coords, {color: '#2563eb', weight: 4, dashArray: '5, 10'}).addTo(globalItineraryMap);
                    }
                } catch (e) {
                    console.error('OSRM routing failed:', e);
                    L.polyline(coords, {color: '#2563eb', weight: 4, dashArray: '5, 10'}).addTo(globalItineraryMap);
                }
            }
            drawRoute(allCoords);
            
            // Add Markers
            itinerary.forEach(day => {
                if (day.detailed_places) {
                    day.detailed_places.forEach((place, idx) => {
                        if (place.lat && place.lon) {
                            const marker = L.circleMarker([place.lat, place.lon], {
                                color: '#fe4fac',
                                fillColor: '#0f172a',
                                fillOpacity: 1,
                                radius: 6,
                                weight: 2
                            }).addTo(globalItineraryMap);
                            marker.bindPopup(`<b>Day ${day.day}</b><br>${place.name}`);
                        }
                    });
                }
            });
        }
    }, 100);
}
