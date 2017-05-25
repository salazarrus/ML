from car import Car
from bicycle import Bicycle
from vehicle import Vehicle


class VehicleFactory:

    def create(vehicleType, vehicle_id):
        return {
            'car' : Car(vehicle_id),
            'bicycle': Bicycle(vehicle_id)
        }[vehicleType]
