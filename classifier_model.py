from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

##############################################################################################
#                                                                                            #
#                                    Classifier Model                                        #
#                                                                                            #
##############################################################################################

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

#Define the list of countries
labels = ['English Name', 'Northern Mariana Islands', 'Kuril Islands', 'France', 'Serbia', 'Uruguay', 'Guam', 'Panama', 'Netherlands Antilles', 'Algeria', 'Togo', "Ma'tan al-Sarra", 
          'Switzerland', 'Jersey', 'Austria', 'Portugal', 'Luxembourg', 'Kazakhstan', 'Aruba', 'Holy See', 'Equatorial Guinea', 'Jamaica', 'Estonia', 'Niger', 'Belize', 'Morocco', 
          'Antigua & Barbuda', 'Norway', 'Bangladesh', 'Denmark', 'Belgium', 'Samoa', 'Anguilla', 'Israel', 'Libyan Arab Jamahiriya', 'Mexico', 'Romania', 'Belarus', 'Ecuador', 'Sudan', 
          'United Republic of Tanzania', 'Micronesia (Federated States of)', 'Malta', 'Finland', 'Turkey', 'Lithuania', 'Russian Federation', 'Zimbabwe', 'Singapore', 'Oman', 
          'Republic of Korea', 'Montserrat', 'Liberia', 'Rwanda', 'Nicaragua', 'Mozambique', 'New Zealand', 'Madeira Islands', 'Kenya', 'Uganda', 'Uzbekistan', 'Ireland', 'Eritrea', 
          'Argentina', 'Congo', 'Bahamas', 'Chile', 'Guinea', 'British Indian Ocean Territory', 'Saint Kitts and Nevis', 'Solomon Islands', 'Pitcairn Island', 'Saint Lucia', 'Turkmenistan', 
          'French Southern and Antarctic Territories', 'Slovenia', 'El Salvador', 'Cook Islands', 'Kuwait', 'Brunei Darussalam', 'Cape Verde', 'Italy', 'Iran (Islamic Republic of)', 
          'United Arab Emirates', 'British Virgin Islands', 'San Marino', 'Zambia', 'Latvia', 'Marshall Islands', 'Syrian Arab Republic', 'Sri Lanka', 'Guantanamo', 'Costa Rica', 
          'Indonesia', 'Liechtenstein', 'Kiribati', 'Fiji', 'Malawi', 'Mayotte', 'Cameroon', 'Hong Kong', 'Yemen', 'Western Sahara', 'Tokelau', 'United States of America', 'Grenada', 
          'Cayman Islands', 'Dominican Republic', 'Glorioso Islands', 'Paracel Islands', 'Guernsey', 'Palau', 'Canada', 'Ethiopia', 'Greenland', 'Comoros', 'Abyei', 'Guatemala', 
          'Kyrgyzstan', 'Suriname', 'Poland', 'Ilemi Triangle', 'Spratly Islands', 'Bermuda', 'Arunachal Pradesh', 'South Sudan', 'Bulgaria', 'U.K. of Great Britain and Northern Ireland', 
          'Bahrain', 'Niue', 'Somalia', 'Barbados', 'Puerto Rico', 'Seychelles', 'Senegal', 'Greece', 'West Bank', 'Nigeria', 'Norfolk Island', 'Aksai Chin', 'Monaco', 'Gambia', 
          'Timor-Leste', 'Honduras', 'Botswana', 'Burkina Faso', 'Ukraine', 'Egypt', 'Germany', "Côte d'Ivoire", 'India', 'Venezuela', 'Bhutan', 'Sierra Leone', 'Tonga', 'Dominica', 
          'Isle of Man', 'Iceland', 'Namibia', '"Moldova, Republic of"', 'Nauru', 'Thailand', 'Nepal', 'American Samoa', 'Midway Is.', 'Reunion', 'Saudi Arabia', 'Taiwan', 'Guadeloupe', 
          'Central African Republic', 'French Polynesia', 'Azores Islands', 'Brazil', 'Haiti', 'Jarvis Island', 'Montenegro', 'Svalbard and Jan Mayen Islands', 'Vietnam', 'Croatia', 
          'Bosnia & Herzegovina', 'Jordan', 'Qatar', 'Tuvalu', 'Saint Vincent and the Grenadines', 'Gaza Strip', 'Hungary', 'Djibouti', 'Vanuatu', 'Cambodia', 'Georgia', 'Christmas Island', 
          'United States Virgin Islands', 'Faroe Islands', 'Lesotho', 'Tajikistan', 'Burundi', 'Philippines', 'Madagascar', 'Azerbaijan', 'Mauritania', 'South Georgia & the South Sandwich Islands', 
          'Mongolia', 'Spain', 'Cyprus', 'Macao', 'Jammu-Kashmir', 'Heard Island and McDonald Islands', 'French Guiana', 'Democratic Republic of the Congo', 'Guinea-Bissau', 'Papua New Guinea', 
          'Swaziland', 'New Caledonia', 'Falkland Islands (Malvinas)', 'Pakistan', 'South Africa', "Democratic People's Republic of Korea", 'Maldives', 'Lebanon', 'Andorra', 'Australia', 
          'Bouvet Island', 'The former Yugoslav Republic of Macedonia', 'Peru', 'Benin', 'Paraguay', 'Turks and Caicos Islands', 'Bolivia', 'Cuba', 'China', 'Iraq', 'Martinique', 'Malaysia', 
          'Sweden', 'Ghana', 'Angola', "Hala'ib Triangle", 'Myanmar', 'Mauritius', 'Cocos (Keeling) Islands', 'Afghanistan', 'Sao Tome and Principe', 'Colombia', 'Czech Republic', 'Japan', 
          "Lao People's Democratic Republic", 'Tunisia', 'Guyana', 'Gabon', 'Netherlands', 'Chad', 'Trinidad and Tobago', 'Slovakia', 'Mali', 'Armenia', 'Albania', 'Gibraltar']

# Classification function
def classify(image):
    """Classify the image to predict the most probable country. Uses pre-defind country labels and image path as parameters.

    Params:
    -------
        image (str):
            Data image path.

    Notes:
    ------
    Uses streetCLIP model based on Open AI's CLIP ViT - https://huggingface.co/geolocal/StreetCLIP.
    """
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    predictions = logits_per_image.softmax(dim=1)

    # Compute classification score for each country
    confidences = {labels[i]: float(predictions[0][i].item()) for i in range(len(labels))}
    return confidences
