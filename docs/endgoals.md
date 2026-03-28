The tool should:
1. Accurately detect usable rooftop area
Identify and exclude obstacles (water tanks, staircases, AC units, etc.)
Output net usable area, not total area
2. Incorporate shadow-aware energy estimation
Detect shadows from nearby buildings, trees, and rooftop objects
Adjust energy output based on time-of-day and seasonal shading
3. Use real-world solar irradiance data
Integrate location-based annual solar datasets
Account for seasonal variation (especially monsoon impact in India)
4. Generate realistic panel placement layouts
Optimize panel count with spacing, tilt, and orientation constraints
Avoid overestimation from ideal packing assumptions
5. Implement physics-based energy modeling
Include losses due to temperature, inverter inefficiency, wiring, and dust
Model long-term degradation of panels
6. Provide accurate financial projections
Use dynamic cost models based on ₹/W (India-specific pricing)
Incorporate electricity tariffs, subsidies, and net metering policies
7. Personalize estimates based on user consumption
Accept household electricity usage input
Compute realistic savings (self-consumption vs grid export)
8. Adapt to Indian rooftop conditions
Specifically handle water tanks, dense urban layouts, and cluttered roofs
Apply dust and maintenance loss factors
9. Improve geospatial and orientation accuracy
Infer roof direction and tilt using satellite/map data
Enhance estimation with location-aware adjustments
10. Provide confidence and transparency metrics
Display confidence score for predictions
Show breakdown of assumptions and potential error margins
11. Continuously improve using ML feedback loop
Learn from real installation data (predicted vs actual)
Refine energy and ROI estimates over time