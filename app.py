from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
import io
import base64
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Earth radius in meters
R = 6_371_000.0

def haversine_dist(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters"""
    φ1, λ1 = np.radians(lat1), np.radians(lon1)
    φ2, λ2 = np.radians(lat2), np.radians(lon2)
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def find_predicted_location(shark_lat, shark_lon):
    """Find the predicted shark location based on eddy currents"""
    data_folder = Path("data/data")
    nc_files = sorted(data_folder.glob("*.nc"))
    
    closest = {"dist": np.inf, "lat": None, "lon": None}
    
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            lat = ds["latitude"].values
            lon = ds["longitude"].values
            lon = ((lon + 180) % 360) - 180
            
            dists = haversine_dist(shark_lat, shark_lon, lat, lon)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = dists[min_idx]
            
            if min_dist < closest["dist"]:
                closest.update({
                    "dist": float(min_dist),
                    "lat": float(lat[min_idx]),
                    "lon": float(lon[min_idx])
                })
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
    
    return closest

def generate_plot(shark_lat, shark_lon, closest):
    """Generate the shark prediction plot"""
    data_folder = Path("data/data")
    nc_files = sorted(data_folder.glob("*.nc"))
    
    dat_folder = Path("data/phytoplankton")
    nc_file = sorted(dat_folder.glob("*.nc"))
    
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    ax.set_title("Shark Location Prediction + Foraging Hotspots", fontsize=14)
    
    # Plot SWOT data
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            lat = ds["latitude"]
            lon = ds["longitude"]
            ssh = ds["ssh_karin"]
            
            lon = ((lon + 180) % 360) - 180
            
            p = ax.pcolormesh(
                lon, lat, ssh,
                transform=ccrs.PlateCarree(),
                alpha=1,
            )
            ds.close()
            
        except Exception as e:
            print(f"Error plotting SWOT {f.name}: {e}")
    
    # Plot phytoplankton data
    for f in nc_file:
        try:
            ds2 = xr.open_dataset(f)
            lat2 = ds2["lat"]
            lon2 = ds2["lon"]
            chlor_a2 = ds2["chlor_a"]
            
            lon2 = ((lon2 + 180) % 360) - 180
            
            p2 = ax.pcolormesh(
                lon2, lat2, chlor_a2,
                transform=ccrs.PlateCarree(),
                alpha=1,
                shading="auto"
            )
            ds2.close()
            
        except Exception as e:
            print(f"Error plotting phytoplankton {f.name}: {e}")
    
    # Plot shark location (black star)
    ax.plot(shark_lon, shark_lat, "k*", markersize=12,
            transform=ccrs.PlateCarree(), label="Shark location")
    
    # Plot predicted location (red dot)
    ax.plot(closest["lon"], closest["lat"], "ro", markersize=10,
            transform=ccrs.PlateCarree(), label="Predicted Shark Location")
    
    ax.legend(loc="lower left")
    plt.colorbar(p, label="Sea Surface Height (m)", orientation="vertical", shrink=0.7)
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Calculate predicted shark location"""
    try:
        data = request.get_json()
        shark_lat = float(data['latitude'])
        shark_lon = float(data['longitude'])
        
        closest = find_predicted_location(shark_lat, shark_lon)
        
        return jsonify({
            'success': True,
            'predicted_lat': closest['lat'],
            'predicted_lon': closest['lon'],
            'distance_km': closest['dist'] / 1000
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Process batch of coordinates from uploaded CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Only CSV files are supported'
            }), 400
        
        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(stream)
        
        results = []
        row_count = 0
        
        for row in csv_reader:
            try:
                # Try different column name variations
                lat = None
                lon = None
                
                # Check for latitude
                for key in row.keys():
                    key_lower = key.lower().strip()
                    if key_lower in ['latitude', 'lat']:
                        lat = float(row[key])
                    if key_lower in ['longitude', 'lon', 'long']:
                        lon = float(row[key])
                
                if lat is None or lon is None:
                    results.append({
                        'row': row_count + 1,
                        'error': 'Missing latitude or longitude',
                        'input_lat': row.get('latitude', row.get('lat', 'N/A')),
                        'input_lon': row.get('longitude', row.get('lon', 'N/A'))
                    })
                    row_count += 1
                    continue
                
                closest = find_predicted_location(lat, lon)
                
                results.append({
                    'row': row_count + 1,
                    'input_lat': lat,
                    'input_lon': lon,
                    'predicted_lat': closest['lat'],
                    'predicted_lon': closest['lon'],
                    'distance_km': closest['dist'] / 1000
                })
                
                row_count += 1
                
            except Exception as e:
                results.append({
                    'row': row_count + 1,
                    'error': str(e),
                    'input_lat': row.get('latitude', row.get('lat', 'N/A')),
                    'input_lon': row.get('longitude', row.get('lon', 'N/A'))
                })
                row_count += 1
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': row_count
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/generate_plot', methods=['POST'])
def plot():
    """Generate and return the plot image as base64"""
    try:
        data = request.get_json()
        shark_lat = float(data['latitude'])
        shark_lon = float(data['longitude'])
        closest = data['closest']
        
        buf = generate_plot(shark_lat, shark_lon, closest)
        
        # Encode to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/download_plot', methods=['POST'])
def download_plot():
    """Generate and download the plot as PNG file"""
    try:
        data = request.get_json()
        shark_lat = float(data['latitude'])
        shark_lon = float(data['longitude'])
        closest = data['closest']
        
        buf = generate_plot(shark_lat, shark_lon, closest)
        
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'shark_prediction_{shark_lat}_{shark_lon}.png'
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)