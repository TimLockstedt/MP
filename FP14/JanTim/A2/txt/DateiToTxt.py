import os

def process_files():
    # Get all files in the current directory
    current_directory = os.getcwd()
    
    # Iterate through all files
    for filename in os.listdir(current_directory):
        # Skip directories
        if os.path.isfile(os.path.join(current_directory, filename)):
            try:
                # Read the file
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Replace commas with periods
                modified_content = content.replace(',', '.')
                
                # Create new filename (ensure it ends with .txt)
                if not filename.lower().endswith('.txt'):
                    new_filename = os.path.splitext(filename)[0] + '.txt'
                else:
                    new_filename = filename
                
                # Write the modified content to a new text file
                with open(new_filename, 'w', encoding='utf-8') as file:
                    file.write(modified_content)
                
                print(f"Processed: {filename} -> {new_filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Run the file processing
if __name__ == "__main__":
    process_files()