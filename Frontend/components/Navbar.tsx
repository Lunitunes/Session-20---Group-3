import Link from "next/link";
import { NavigationMenu, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, navigationMenuTriggerStyle } from "./ui/navigation-menu";
import { ThemeToggle } from "./ThemeToggle";


export default function Navbar() {
    return(
        <nav className="sticky w-auto m-auto mt-4">
            <NavigationMenu className="m-auto border rounded-full p-2 px-4">
                <NavigationMenuList className="flex-wrap gap-3 justify-between w-xl">

                    <NavigationMenuItem>
                        <NavigationMenuLink asChild className={navigationMenuTriggerStyle()}>
                            <Link href="/">Home</Link>
                        </NavigationMenuLink>
                    </NavigationMenuItem>

                    <NavigationMenuItem>
                        <NavigationMenuLink asChild className={navigationMenuTriggerStyle()}>
                            <Link href="/input">Input</Link>
                        </NavigationMenuLink>
                    </NavigationMenuItem>

                    <NavigationMenuItem>
                        <NavigationMenuLink asChild className={navigationMenuTriggerStyle()}>
                            <Link href="/data-visualisations">Data Visualisatoins</Link>
                        </NavigationMenuLink>
                    </NavigationMenuItem>

                    <NavigationMenuItem>
                        <NavigationMenuLink asChild className={navigationMenuTriggerStyle()}>
                            <Link href="/about">About</Link>
                        </NavigationMenuLink>
                    </NavigationMenuItem>

                    <NavigationMenuItem>
                        <ThemeToggle/>
                    </NavigationMenuItem>
                    
                </NavigationMenuList>
            </NavigationMenu>
        </nav>
    )
}