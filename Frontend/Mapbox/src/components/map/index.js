import React, { useEffect } from 'react'
import axios from 'axios'
import 'mapbox-gl/dist/mapbox-gl.css'
import ReactMapboxGl, { Source } from 'react-mapbox-gl'

// const url = 'https://dbscanmap.herokuapp.com'
const url = 'http://localhost:5000'

const Map = ReactMapboxGl({
  accessToken: 'pk.eyJ1IjoiYnVpYmFhbmgiLCJhIjoiY2tvc2J5NGpnMDAzMTJvbnA5c25oa3h5YyJ9.-Z68HDVZedI-SwAbgKR-lw',
})

const RASTER_SOURCE_OPTIONS = {
  type: 'geojson',
  data: url + '/window',
  // cluster: true,
  // clusterMaxZoom: 14, // Max zoom to cluster points on
  // clusterRadius: 50,
}

const load = (map) => {
  map.addLayer({
    id: 'population',
    type: 'circle',
    source: 'data',
    paint: {
      // make circles larger as the user zooms from z12 to z22
      'circle-radius': {
        base: 1.75,
        stops: [
          [12, 2],
          [22, 180],
        ],
      },
      // color circles by ethnicity, using a match expression
      // https://docs.mapbox.com/mapbox-gl-js/style-spec/#expressions-match
      'circle-color': [
        'match',
        ['get', 'clusterid'],
	-1,
	'#4a55b0',
	0,
	'#1e11cc',
	1,
	'#ed0c57',
	2,
	'#b635a4',
	3,
	'#f899f1',
	4,
	'#3a102d',
	5,
	'#0b5494',
	6,
	'#2e0c4c',
	7,
	'#14615e',
	8,
	'#a9ee7a',
	9,
	'#9d03c3',
	10,
	'#92e10d',
	11,
	'#badd55',
	12,
	'#37448d',
	13,
	'#410052',
	14,
	'#dcd9f6',
	15,
	'#fd21d1',
	16,
	'#b9024f',
	17,
	'#F48FB1',
	18,
	'#177d7b',
	19,
	'#327265',
	20,
	'#4df724',
	21,
	'#D81B60',
	22,
	'#C2185B',
	23,
	'#AD1457',
	24,
	'#880E4F',
	25,
	'#FF80AB',
	26,
	'#FF4081',
	'#8B00FF',

      ],
    },
  })

  window.setInterval(async () => {
    const result = await axios.get(url+'/window')
    if (result.status === 200) {
      const json = result.data
      map.getSource('data').setData(json)
    }
  }, 10000)
}

const MapBox = () => {
  return (
    <Map
      style="mapbox://styles/mapbox/light-v10"
      containerStyle={{
        height: '100vh',
        width: '100vw',
      }}
      center={[-87.6451618298, 41.7941646462]}
      zoom={[12]}	
      onStyleLoad={(map, _) => load(map)}
    >
      <Source id="data" geoJsonSource={RASTER_SOURCE_OPTIONS} />
    </Map>
  )
}

export default MapBox
