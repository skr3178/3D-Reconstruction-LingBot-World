import streetview

panoid = "mwwbD0i1n2MRibtZJu7UXA"

# zoom=5 is max resolution (~13312x6656)
image = streetview.get_panorama(panoid, zoom=5)
image.save("output.jpg")