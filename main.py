import pygame
import math

# Initialize Pygame
pygame.init()

# Image directory
IMAGE_DIR = "resources/images"

# Screen Settings
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Airport with Aircraft Landing")

# Colors
LIGHT_GREEN = (19, 105, 42)  # Ground
GRAY = (105, 105, 105)  # Runway
WHITE = (255, 255, 255)  # Runway Divider
YELLOW = (255, 215, 0)  # Taxiway Middle Line
DARK_GRAY = (60, 60, 60)  # Taxiways
BLUE = (176, 118, 2)  # Buildings

# Frame Rate
clock = pygame.time.Clock()

# Load Aircraft Image
aircraft_img = pygame.image.load(f"{IMAGE_DIR}/aircraft.png")  
aircraft_img = pygame.transform.scale(aircraft_img, (100, 100))  # Resize to 100x100
# Create a rotated version for descent
aircraft_img_descent = pygame.transform.rotate(aircraft_img, -10)  # Slight nose-down angle for descent

# Runway Settings
runway_width = 60
runway_spacing = 200  # Restored to original spacing
taxiway_width = 40

# Runway Positions - using original positions
runway_x1 = 100
runway_x2 = 100
runway_x3 = 200  # Third runway shifted right

runway_y_positions = [
    150,  # Top runway
    150 + runway_spacing,  # Middle runway
    150 + 2 * runway_spacing  # Bottom runway
]

# Buildings between runways
building_width = 150
building_height = 100
building_x = 500

buildings = [
    pygame.Rect(building_x, runway_y_positions[0] + runway_width + 20, building_width, building_height),
    pygame.Rect(building_x, runway_y_positions[1] + runway_width + 20, building_width, building_height)
]

# Taxiway Roads (Properly Connecting Runways & Buildings)
taxiways = [
    # Vertical taxiways connecting first and second runway
    pygame.Rect(200, runway_y_positions[0] + runway_width, taxiway_width, runway_spacing - runway_width),  # Left vertical taxiway
    pygame.Rect(700, runway_y_positions[0] + runway_width, taxiway_width, runway_spacing - runway_width),  # Right vertical taxiway

    # Vertical taxiways connecting second and third runway
    pygame.Rect(300, runway_y_positions[1] + runway_width, taxiway_width, runway_spacing - runway_width),  # Left vertical taxiway
    pygame.Rect(800, runway_y_positions[1] + runway_width, taxiway_width, runway_spacing - runway_width),  # Right vertical taxiway

    # Horizontal taxiways connecting buildings to vertical taxiways
    pygame.Rect(200 + taxiway_width, runway_y_positions[0] + runway_width + (building_height // 2), building_x - (200 + taxiway_width), taxiway_width),  # Top front building taxiway
    pygame.Rect(300 + taxiway_width, runway_y_positions[1] + runway_width + (building_height // 2), building_x - (300 + taxiway_width), taxiway_width),   # Bottom front building taxiway
    pygame.Rect(building_x + building_width, runway_y_positions[0] + runway_width + (building_height // 2), 700 - (building_x + building_width), taxiway_width),  # Top back building taxiway
    pygame.Rect(building_x + building_width, runway_y_positions[1] + runway_width + (building_height // 2), 800 - (building_x + building_width), taxiway_width)  # Bottom back building taxiway
]

# Aircraft Settings for each runway
aircraft_width, aircraft_height = aircraft_img.get_size()

# Create three aircraft - one for each runway
aircraft = [
    {
        "x": -300,
        "y": 50,
        "target_y": runway_y_positions[0] + (runway_width // 2) - (aircraft_height // 2),
        "is_descending": True,
        "is_on_ground": False,
        "speed": 5,
        "runway_index": 0
    },
    {
        "x": -600,  # Staggered starting position
        "y": 30,
        "target_y": runway_y_positions[1] + (runway_width // 2) - (aircraft_height // 2),
        "is_descending": True,
        "is_on_ground": False,
        "speed": 5.5,  # Slightly different speed
        "runway_index": 1
    },
    {
        "x": -900,  # Staggered starting position
        "y": 70,
        "target_y": runway_y_positions[2] + (runway_width // 2) - (aircraft_height // 2),
        "is_descending": True,
        "is_on_ground": False,
        "speed": 4.5,  # Slightly different speed
        "runway_index": 2
    }
]

# Landing parameters
initial_altitude = 50
descent_distance = 800  # Horizontal distance over which descent occurs
touchdown_x = 600  # Where the aircraft fully touches down
stopped_x = 800  # Where the aircraft stops

# Main Game Loop
running = True
while running:
    # Fill background with ground color
    screen.fill(LIGHT_GREEN)

    # Draw Runways
    pygame.draw.rect(screen, GRAY, (runway_x1, runway_y_positions[0], 900, runway_width))  # Runway 1
    pygame.draw.rect(screen, GRAY, (runway_x2, runway_y_positions[1], 900, runway_width))  # Runway 2
    pygame.draw.rect(screen, GRAY, (runway_x3, runway_y_positions[2], 900, runway_width))  # Runway 3 (shifted)

    # Draw Dashed Runway Divider
    for i in range(100, 1000, 40):
        pygame.draw.line(screen, WHITE, (i, runway_y_positions[0] + (runway_width // 2)), (i + 20, runway_y_positions[0] + (runway_width // 2)), 4)
        pygame.draw.line(screen, WHITE, (i, runway_y_positions[1] + (runway_width // 2)), (i + 20, runway_y_positions[1] + (runway_width // 2)), 4)
        pygame.draw.line(screen, WHITE, (i + 100, runway_y_positions[2] + (runway_width // 2)), (i + 120, runway_y_positions[2] + (runway_width // 2)), 4)  # Shifted for 3rd runway

    # Draw Buildings
    for building in buildings:
        pygame.draw.rect(screen, BLUE, building)

    # Draw Taxiways
    for taxiway in taxiways:
        pygame.draw.rect(screen, DARK_GRAY, taxiway)

    # Draw Taxiway Lines (Yellow) - Vertical
    for taxiway in taxiways[:4]:  # First four taxiways are vertical
        pygame.draw.line(screen, YELLOW, (taxiway.x + taxiway.width // 2, taxiway.y), 
                         (taxiway.x + taxiway.width // 2, taxiway.y + taxiway.height), 4)

    # Draw Taxiway Lines (Yellow) - Horizontal
    for taxiway in taxiways[4:]:  # Last four taxiways are horizontal
        pygame.draw.line(screen, YELLOW, (taxiway.x, taxiway.y + taxiway.height // 2), 
                         (taxiway.x + taxiway.width, taxiway.y + taxiway.height // 2), 4)

    # Update and draw each aircraft
    for plane in aircraft:
        # Move aircraft forward
        plane["x"] += plane["speed"]

        # Handle descent for each aircraft
        if plane["is_descending"]:
            # Calculate descent progress (0.0 to 1.0)
            descent_progress = min(1.0, max(0.0, (plane["x"] + 300) / descent_distance))
            
            # Use a smooth curve for descent (ease-in-out)
            smooth_progress = 0.5 - 0.5 * math.cos(descent_progress * math.pi)
            plane["y"] = initial_altitude + smooth_progress * (plane["target_y"] - initial_altitude)
            
            # Draw descending aircraft (angled)
            if plane["x"] > -100:  # Only draw when partially visible
                screen.blit(aircraft_img_descent, (plane["x"], plane["y"]))
            
            # Check if touchdown
            if plane["x"] >= touchdown_x:
                plane["is_descending"] = False
                plane["is_on_ground"] = True
                plane["y"] = plane["target_y"]  # Ensure exact alignment with runway
        
        elif plane["is_on_ground"]:
            # Draw level aircraft
            screen.blit(aircraft_img, (plane["x"], plane["y"]))
            
            # Slow down after touchdown
            if plane["x"] < stopped_x:
                plane["speed"] = max(1, plane["speed"] * 0.98)  # Gradual deceleration
            else:
                plane["speed"] = 0  # Stop completely
        
        # Reset aircraft when it goes off the right edge
        if plane["x"] > WIDTH + 100:
            plane["x"] = -300 - (plane["runway_index"] * 200)  # Stagger reset positions
            plane["y"] = 30 + (plane["runway_index"] * 20)  # Vary starting heights
            plane["speed"] = 4.5 + (plane["runway_index"] * 0.5)  # Vary speeds slightly
            plane["is_descending"] = True
            plane["is_on_ground"] = False

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)

# Quit Pygame
pygame.quit()