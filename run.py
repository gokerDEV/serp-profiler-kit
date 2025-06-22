import os
import subprocess

def list_subscripts():
    return {
        # Data Collection
        'Website Acquisition (Google)': {
            'script': 'src/data_collection/search_on_google.py',
            'args': ['--input', '--output', '--max_page', '--overwrite', '--exceptions'],
            'defaults': ['data/raw/keywords/keywords.csv', 'data/raw/search_results/google', '2', '0', 'data/exceptions/google-search.csv']
        },
        'Website Acquisition (Bing)': {
            'script': 'src/data_collection/search_on_bing.py',
            'args': ['--input', '--output', '--max_page', '--overwrite', '--exceptions'],
            'defaults': ['data/raw/keywords/keywords.csv', 'data/raw/search_results/bing', '1', '0', 'data/exceptions/bing-search.csv']
        },
        'Dump SERPs (Google & Bing)': {
            'script': 'src/data_processing/dump_serps.py',
            'args': ['--input', '--output', '--invalids'],
            'defaults': ['data/raw/keywords/keywords.csv', 'data/processed/serps.csv', 'data/exceptions/serps.csv']
        },
        
        # Data Processing
        'Scraper': {
            'script': 'src/data_collection/scraper.py',
            'args': ['--overwrite', '--skip-exceptions', '--input', '--output', '--exceptions', '--browser', '--headless', '--screenshots'],
            'defaults': ['0', '1', 'data/processed/serps.csv', 'data/raw/pages/', 'data/exceptions/scraper.csv', 'chromium', '1', 'data/raw/screenshots/']
        },
        'Acceptance': {
            'script': 'src/data_processing/acceptance.py',
            'args': ['--input', '--input_dir', '--output', '--exceptions'],
            'defaults': ['data/processed/serps.csv', 'data/raw/', 'data/processed/accepted.csv', 'data/exceptions/acceptance.csv']
        },
        'Feature Extraction': {
            'script': 'src/data_processing/feature_extraction.py',
            'args': ['--input', '--raw_dir', '--output', '--model'],
            'defaults': ['data/processed/accepted.csv', 'data/raw', 'data/processed/features.csv', 'all-mpnet-base-v2']
        },
        
        'Pagespeed': {
            'script': 'src/data_collection/get_pagespeed.py',
            'args': ['--input', '--output', '--exceptions'],
            'defaults': ['data/processed/accepted.csv', 'data/raw/pagespeed', 'data/exceptions/pagespeed.csv']
        },
        'Dump Pagespeed': {
            'script': 'src/data_processing/dump_pagespeed.py',
            'args': ['--input', '--output', '--exceptions'],
            'defaults': ['data/raw/pagespeed', 'data/processed/pagespeed.csv', 'data/exceptions/pagespeed_dump.csv']
        },
        'Combine': {
            'script': 'src/data_processing/combine.py',
            'args': ['--features', '--pagespeed', '--output', '--exceptions'],
            'defaults': ['data/processed/features.csv', 'data/processed/pagespeed.csv', 'data/processed/combined.csv', 'data/exceptions/combine.csv']
        },
        'Outliers': {
            'script': 'src/data_processing/outliers.py',
            'args': ['--input', '--output'],
            'defaults': ['data/processed/combined.csv', 'data/outliers']
        },
        'Create Dataset': {
            'script': 'src/data_processing/dataset.py',
            'args': ['--input', '--output'],
            'defaults': ['data/processed/combined.csv', 'data/datasets/dataset.csv']
        },
        
        'Dataset Info': {
            'script': 'src/data_processing/info.py',
            'args': ['--input', '--output'],
            'defaults': ['data/datasets/dataset.csv', 'data/info']
        },
        
        # Analysis
        'Analyze': {
            'script': 'src/analysis/analyze.py',
            'args': ['--input', '--output'],
            'defaults': ['data/datasets/dataset.csv', 'data/analysis']
        },

        
        # Generation
        'Plot Generation': {
            'script': 'src/generators/plot.py',
            'args': ['--results_path', '--dataset_path', '--output_dir'],
            'defaults': ['data/analysis/analysis_results.json', 'data/analysis/dataset_with_clusters.csv', 'data/analysis/plots']
        },
           
        'LaTeX Table Generation': {
            'script': 'src/generators/table.py',
            'args': ['--results_path', '--output_dir'],
            'defaults': ['data/analysis/analysis_results.json', 'data/analysis/tables']
        },
        
    }

def main():
    subscripts = list_subscripts()
    
    # Display the available scripts with their indices
    print("Available scripts:")
    print("=" * 80)
    
    current_category = None
    for index, (name, details) in enumerate(subscripts.items(), start=1):
        # Determine category based on script path
        script_path = details['script']
        if 'data_collection' in script_path:
            category = "📊 Data Collection"
        elif 'data_processing' in script_path:
            category = "🔄 Data Processing"
        elif 'analysis' in script_path:
            if 'Legacy' in name:
                category = "📈 Analysis (Legacy)"
            else:
                category = "📈 Analysis (Modular)"
        elif 'generators' in script_path:
            category = "📋 Generation"
        else:
            category = "🔧 Other"
        
        # Print category header if it's a new category
        if category != current_category:
            print(f"\n{category}:")
            print("-" * 40)
            current_category = category
        
        print(f"{str(index).ljust(2)} - {name}")
    
    print("\n" + "=" * 80)
    
    # Get user input for script selection
    choice = input("Enter the number of the script you want to run: ")
    
    # Validate the choice and run the selected script
    try:
        choice_index = int(choice) - 1  # Convert to zero-based index
        if 0 <= choice_index < len(subscripts):
            name = list(subscripts.keys())[choice_index]  # Get the name based on index
            details = subscripts[name]  # Access the details using the name
            script_path = details['script']
            args = details['args']
            defaults = details['defaults']
            
            print(f"\nSelected: {name}")
            print(f"Script: {script_path}")
            print(f"Default arguments: {dict(zip(args, defaults))}")
            
            # Ask if the user wants to edit the arguments
            edit_args = input("\nDo you want to change the default arguments? (Press 'e' to edit, any other key to continue): ")
            if edit_args.lower() == 'e':
                # Prepare to collect arguments for the selected sub-script
                script_args = {}
                skip_all = False
                
                for i, arg in enumerate(args):
                    default_value = defaults[i]
                    user_input = input(f"Enter value for {arg} (or press Enter to use default '{default_value}'): ")
                    
                    if user_input == '':
                        script_args[arg] = default_value  # Use default value
                    else:
                        script_args[arg] = user_input
            
            else:
                # Use default values for all arguments
                script_args = {arg: default for arg, default in zip(args, defaults)}
            
            # Prepare the command to run the selected sub-script with its arguments
            command = ['python', script_path]
            for i, arg in enumerate(args):
                command.append(arg)  # Append the argument name
                # Skip empty values for boolean flags
                if script_args[arg] != '':
                    command.append(script_args[arg])  # Append the argument value

            print(f"\nExecuting: {' '.join(command)}")
            print("=" * 80)
            
            # Run the selected sub-script
            subprocess.run(command)  # Execute the command
        else:
            print("Invalid selection. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()