from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv('women_in_stem.csv')
print(f"Loaded {len(df)} records")

# Home page
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to Women in STEM API",
        "total_records": len(df)
    })

# Get all data
@app.route('/data')
def get_data():
    limit = request.args.get('limit', type=int)
    
    if limit:
        result = df.head(limit)
    else:
        result = df
    
    return jsonify({
        "count": len(result),
        "data": result.to_dict('records')
    })

# Simple search
@app.route('/search')
def search():
    result = df.copy()
    
    country = request.args.get('country')
    if country:
        result = result[result['Country'].str.contains(country, case=False)]
    
    field = request.args.get('field')
    if field:
        result = result[result['STEM Fields'].str.contains(field, case=False)]
    
    year = request.args.get('year', type=int)
    if year:
        result = result[result['Year'] == year]
    
    return jsonify({
        "found": len(result),
        "data": result.to_dict('records')
    })

# Advanced filtering
@app.route('/apply', methods=['POST'])
def apply():
    filters = request.json
    result = df.copy()
    
    if 'countries' in filters:
        result = result[result['Country'].isin(filters['countries'])]
    
    if 'fields' in filters:
        result = result[result['STEM Fields'].isin(filters['fields'])]
    
    if 'year_range' in filters:
        if 'start' in filters['year_range']:
            result = result[result['Year'] >= filters['year_range']['start']]
        if 'end' in filters['year_range']:
            result = result[result['Year'] <= filters['year_range']['end']]
    
    if 'graduation_threshold' in filters:
        result = result[result['Female Graduation Rate (%)'] >= filters['graduation_threshold']]
    
    if 'sort_by' in filters:
        asc = filters.get('order', 'asc') == 'asc'
        result = result.sort_values(filters['sort_by'], ascending=asc)
    
    if 'limit' in filters:
        result = result.head(filters['limit'])
    
    return jsonify({
        "total": len(result),
        "data": result.to_dict('records')
    })

# List of countries
@app.route('/countries')
def countries():
    return jsonify({
        "countries": df['Country'].unique().tolist()
    })

# List of fields
@app.route('/fields')
def fields():
    return jsonify({
        "fields": df['STEM Fields'].unique().tolist()
    })

if __name__ == '__main__':
    print("Server running on http://localhost:5000")
    app.run(debug=True)